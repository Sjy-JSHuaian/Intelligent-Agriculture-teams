"""语义分割指标：整体 Accuracy + 类别平均 Precision/Recall/IoU/Dice。

基于混淆矩阵在整个验证集上累计，再按类计算，忽略 ignore_index 像素。
"""
from __future__ import annotations

import numpy as np


class SegmentationMetrics:
    """累计混淆矩阵并计算 mIoU / mDice / mPrecision / mRecall / Accuracy。"""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_correct = 0
        self.total_valid = 0

    def update(self, pred: "np.ndarray | object", target: "np.ndarray | object") -> None:
        """累计一批预测结果。

        Args:
            pred: 预测类别，形状 (N, H, W) 的 torch.Tensor 或 numpy 数组。
            target: 真实标签，形状相同。
        """
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        if hasattr(target, "detach"):
            target = target.detach().cpu().numpy()
        pred = np.asarray(pred).ravel()
        target = np.asarray(target).ravel()
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        self.total_valid += int(valid.sum())
        self.total_correct += int((pred == target).sum())
        for t in range(self.num_classes):
            tmask = target == t
            if not tmask.any():
                continue
            for p in range(self.num_classes):
                self.cm[t, p] += int((tmask & (pred == p)).sum())

    def compute(self) -> dict:
        """返回整体 Accuracy 与类别平均 Precision/Recall/IoU/Dice。"""
        eps = 1e-7
        ious, dices, precisions, recalls = [], [], [], []
        for c in range(self.num_classes):
            tp = float(self.cm[c, c])
            fp = float(self.cm[:, c].sum() - tp)
            fn = float(self.cm[c, :].sum() - tp)
            precisions.append(tp / (tp + fp + eps))
            recalls.append(tp / (tp + fn + eps))
            ious.append(tp / (tp + fp + fn + eps))
            dices.append(2.0 * tp / (2.0 * tp + fp + fn + eps))
        return {
            "accuracy": float(self.total_correct) / max(1, self.total_valid),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "iou": float(np.mean(ious)),
            "dice": float(np.mean(dices)),
        }
