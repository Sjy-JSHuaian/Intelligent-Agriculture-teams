"""损失函数：交叉熵 + soft Dice 的组合损失。"""
from __future__ import annotations

import torch
import torch.nn as nn


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   ignore_index: int = 255, smooth: float = 1.0) -> torch.Tensor:
    """多类 soft Dice 损失，忽略 ignore_index 像素。

    Args:
        logits: (N, C, H, W) 未经过 softmax 的 logits。
        target: (N, H, W) 整数类别标签，ignore_index 像素不参与。
    """
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    valid = target != ignore_index
    dice_sum = 0.0
    n = 0
    for c in range(num_classes):
        t = ((target == c) & valid).float()
        if t.sum() == 0:
            continue
        p = probs[:, c]
        inter = (p * t).sum()
        union = p[valid].sum() + t.sum()
        dice_sum += (2.0 * inter + smooth) / (union + smooth)
        n += 1
    if n == 0:
        return torch.tensor(0.0, device=logits.device)
    return 1.0 - dice_sum / n


class CombinedLoss(nn.Module):
    """CrossEntropy(ignore=255) + dice_weight * soft Dice。"""

    def __init__(self, ignore_index: int = 255, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice = soft_dice_loss(logits, target, ignore_index=self.ce.ignore_index)
        return ce + self.dice_weight * dice
