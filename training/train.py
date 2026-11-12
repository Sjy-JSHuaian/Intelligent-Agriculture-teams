"""U-Net 基线训练脚本。

用法（在 image_analysis 根目录运行）：
    python training/train.py --config configs/train.yaml

调试可只跑若干步：python training/train.py --max_steps 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.augmentation import get_train_augmentation, get_val_augmentation
from datasets.dataset_loader import (IGNORE_INDEX, NUM_CLASSES, LoveDADataset,
                                     collect_samples)
from models import build_model
from training.losses import CombinedLoss
from training.metrics import SegmentationMetrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(cfg: dict):
    root = str(ROOT / cfg["data"]["root"])
    scenes = cfg["data"]["scenes"]
    tile_size = (cfg["data"]["tile_size"], cfg["data"]["tile_size"])
    seed = cfg["training"]["seed"]

    # 训练子集：打乱后取前 N（保证 urban/rural 混合，用固定 seed 可复现）
    train_pairs = collect_samples(root, "train", scenes)
    rng = np.random.default_rng(seed)
    rng.shuffle(train_pairs)
    train_pairs = train_pairs[: cfg["data"]["train_subset"]]

    val_pairs = collect_samples(root, "val", scenes)[: cfg["data"]["val_subset"]]

    train_ds = LoveDADataset(train_pairs, get_train_augmentation(tile_size), tile_size)
    val_ds = LoveDADataset(val_pairs, get_val_augmentation(tile_size), tile_size)

    nw = cfg["training"]["num_workers"]
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader


def validate(model, loader, criterion, device) -> tuple[float, dict]:
    model.eval()
    metrics = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX)
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            metrics.update(logits.argmax(dim=1), y)
    val_loss /= max(1, len(loader.dataset))
    return val_loss, metrics.compute()


def plot_curves(history: dict, save_path: Path) -> None:
    """保存 loss 与指标曲线图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    m = history["val_metrics"]
    for key, label in [("iou", "mIoU"), ("dice", "mDice"), ("accuracy", "Acc")]:
        axes[1].plot(epochs, [v[key] for v in m], label=label)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].set_title("Validation metrics"); axes[1].legend(); axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--max_steps", type=int, default=None, help="调试用：只跑 N 步")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / args.model_config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    model_name = model_cfg.get("name", "unet")

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch_threads={torch.get_num_threads()}")

    train_loader, val_loader = build_loaders(cfg)
    print(f"train 样本: {len(train_loader.dataset)}, val 样本: {len(val_loader.dataset)}")

    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M")

    criterion = CombinedLoss(ignore_index=IGNORE_INDEX,
                             dice_weight=cfg["training"]["dice_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"],
                                 weight_decay=cfg["training"]["weight_decay"])

    save_dir = ROOT / cfg["training"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_dice = -1.0
    global_step = 0
    n_epochs = cfg["training"]["epochs"]
    log_interval = cfg["training"]["log_interval"]

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_steps and global_step >= args.max_steps:
                break

        epoch_loss /= max(1, n_batches)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        print(f"[Epoch {epoch}] train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} "
              f"mIoU={val_metrics['iou']:.4f} mDice={val_metrics['dice']:.4f} "
              f"Acc={val_metrics['accuracy']:.4f} ({time.time() - t0:.0f}s)")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "metrics": val_metrics}, save_dir / f"{model_name}_best.pth")
            print(f"  -> 保存最佳模型 (mDice={best_dice:.4f})")

        if args.max_steps and global_step >= args.max_steps:
            break

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    plot_curves(history, save_dir / "training_curves.png")
    print(f"训练结束，最佳 mDice={best_dice:.4f}，历史保存于 {save_dir}")


if __name__ == "__main__":
    main()
