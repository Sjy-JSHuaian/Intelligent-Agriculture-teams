"""模型评估与对比：在验证集上计算指标 + 参数量 + 推理耗时 + 模型体积。

用法：
    python scripts/evaluate.py --compare configs/compare.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# SegFormer 预训练权重已缓存；设镜像避免联网（国内环境）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.augmentation import get_val_augmentation
from datasets.dataset_loader import (IGNORE_INDEX, NUM_CLASSES, LoveDADataset,
                                     collect_samples)
from models import build_model
from training.metrics import SegmentationMetrics


def load_model(model_cfg: dict, checkpoint: Path):
    model = build_model(model_cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_model(model, val_loader) -> dict:
    metrics = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX)
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["image"])
            metrics.update(logits.argmax(1), batch["mask"])
    return metrics.compute()


def measure_inference_ms(model, sample_input, repeats: int = 20, warmup: int = 3) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(sample_input)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(sample_input)
        return (time.perf_counter() - t0) / repeats * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", default="configs/compare.yaml")
    args = parser.parse_args()

    with open(ROOT / args.compare, encoding="utf-8") as f:
        cmp = yaml.safe_load(f)

    data_cfg = cmp["data"]
    root = str(ROOT / data_cfg["root"])
    scenes = data_cfg["scenes"]
    tile_size = (data_cfg["tile_size"], data_cfg["tile_size"])

    val_pairs = collect_samples(root, "val", scenes)[: data_cfg["val_subset"]]
    val_ds = LoveDADataset(val_pairs, get_val_augmentation(tile_size), tile_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    sample_input = val_ds[0]["image"].unsqueeze(0)   # (1, C, H, W)

    results = []
    for spec in cmp["models"]:
        name = spec["name"]
        with open(ROOT / spec["model_config"], encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)["model"]
        checkpoint = ROOT / spec["checkpoint"]
        model, ckpt = load_model(model_cfg, checkpoint)

        params = sum(p.numel() for p in model.parameters())
        metrics = evaluate_model(model, val_loader)
        inf_ms = measure_inference_ms(model, sample_input)
        model_size_mb = checkpoint.stat().st_size / 1e6

        row = {
            "name": name,
            "epoch": ckpt.get("epoch"),
            "params_m": round(params / 1e6, 2),
            "params_memory_mb": round(params * 4 / 1e6, 1),  # fp32 权重内存估算
            "model_size_mb": round(model_size_mb, 1),
            "inference_ms": round(inf_ms, 2),
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "iou": round(metrics["iou"], 4),
            "dice": round(metrics["dice"], 4),
        }
        results.append(row)
        print(f"[{name}] epoch={row['epoch']} params={row['params_m']}M "
              f"infer={row['inference_ms']}ms mIoU={row['iou']} mDice={row['dice']} "
              f"Acc={row['accuracy']}")

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n| 模型 | 参数(M) | 权重内存(MB) | 推理(ms) | Acc | Prec | Recall | mIoU | mDice |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['params_m']} | {r['params_memory_mb']} | {r['inference_ms']} | "
              f"{r['accuracy']} | {r['precision']} | {r['recall']} | {r['iou']} | {r['dice']} |")
    print(f"\n结果已保存到 {out / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
