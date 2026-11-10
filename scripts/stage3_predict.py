"""阶段 3 分割结果预测与可视化。

加载训练好的 U-Net 最优 checkpoint，对若干验证样本预测并输出：
  原始图 / 真值掩膜 / 预测掩膜 / 叠加图 / 逐图指标(Acc/IoU/Dice) / 类别面积分布。

用法：
    python scripts/stage3_predict.py --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.augmentation import get_val_augmentation
from datasets.dataset_loader import (IGNORE_INDEX, NUM_CLASSES, TRAIN_CLASSES,
                                     collect_samples, remap_mask)
from datasets.preprocessing import read_image, read_mask
from models.unet import build_unet
from training.metrics import SegmentationMetrics
from visualization.segmentation_visualizer import save_color_mask, save_overlay


def predict_one(model, image: np.ndarray, val_aug) -> np.ndarray:
    """对单张 HWC uint8 影像预测，返回 (H, W) 预测类别（0-6）。"""
    inp = val_aug(image=image)["image"]           # HWC float 256x256
    inp = np.ascontiguousarray(inp.transpose(2, 0, 1))  # CHW
    x = torch.from_numpy(inp).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(1).squeeze(0).numpy()    # (256, 256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pth")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data", default="data/raw/loveda")
    parser.add_argument("--scenes", nargs="+", default=["urban", "rural"])
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default="outputs/segmentation")
    args = parser.parse_args()

    with open(ROOT / args.config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["model"]
    model_cfg.pop("name", None)

    model = build_unet(**model_cfg)
    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"已加载 checkpoint: {args.checkpoint}（epoch={ckpt.get('epoch')}，"
          f"mDice={ckpt['metrics']['dice']:.4f}）")

    val_aug = get_val_augmentation((256, 256))
    pairs = collect_samples(str(ROOT / args.data), "val", args.scenes)[: args.n]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for img_path, mask_path in pairs:
        image = read_image(img_path)               # HWC uint8 RGB
        raw_mask = read_mask(mask_path)            # HW 0-7
        H, W = image.shape[:2]
        name = Path(img_path).stem

        pred = predict_one(model, image, val_aug)  # 256x256, 0-6

        # 真值在 256 分辨率上的重映射（用于计算指标）
        gt_256 = cv2.resize(raw_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_remap = remap_mask(gt_256)              # 0-6, 255=ignore

        # 指标（256 分辨率预测 vs 真值）
        m = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX)
        m.update(pred, gt_remap)
        metrics = m.compute()

        # 类别面积分布（在原始分辨率上，预测上采样回原尺寸）
        pred_full = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        dist = {TRAIN_CLASSES[c]: float((pred_full == c).sum()) for c in range(NUM_CLASSES)}
        total = sum(dist.values())
        dist_pct = {k: round(v / total * 100, 2) if total else 0.0 for k, v in dist.items()}

        # 预测 0-6 -> 原始 1-7 配色
        pred_raw = pred_full.astype(np.int64) + 1

        save_overlay(image, pred_raw, str(out_dir / f"{name}_overlay.png"))
        save_color_mask(raw_mask, str(out_dir / f"{name}_gt.png"))
        save_color_mask(pred_raw, str(out_dir / f"{name}_pred.png"))
        ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        buf.tofile(str(out_dir / f"{name}_image.png"))

        print(f"[{name}] Acc={metrics['accuracy']:.4f} mIoU={metrics['iou']:.4f} "
              f"mDice={metrics['dice']:.4f}")
        summary.append({"name": name, "metrics": metrics, "class_distribution_pct": dist_pct})

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果与指标已保存到 {out_dir}")


if __name__ == "__main__":
    main()
