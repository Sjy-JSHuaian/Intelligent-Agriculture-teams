"""SegFormer 模型（基于 transformers 的 MiT-B0 预训练骨干）。

SegFormer-B0 只有 ~3.7M 参数，比 ResNet18 U-Net(14.3M) 还轻，适合 CPU。
ImageNet 预训练的 MiT 骨干 + 随机初始化的 7 类解码头（迁移学习）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# 与 dataset_loader.TRAIN_CLASSES 一致的标签映射（重映射后 0-6）
LOVEDA_ID2LABEL = {
    0: "background", 1: "building", 2: "road", 3: "water",
    4: "barren", 5: "forest", 6: "agriculture",
}


class SegFormerWrapper(nn.Module):
    """包装 transformers SegFormer，输出与输入同分辨率的 logits，兼容现有训练循环。"""

    def __init__(self, num_classes: int = 7, pretrained: str = "nvidia/mit-b0"):
        super().__init__()
        self.id2label = LOVEDA_ID2LABEL
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits                       # (B, C, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


def build_segformer(num_classes: int = 7, pretrained: str = "nvidia/mit-b0", **kwargs):
    """构建 SegFormer（ImageNet 预训练骨干 + 7 类随机解码头）。"""
    return SegFormerWrapper(num_classes=num_classes, pretrained=pretrained)
