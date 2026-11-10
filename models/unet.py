"""U-Net 模型（基于 segmentation_models_pytorch）。"""
from __future__ import annotations

import segmentation_models_pytorch as smp


def build_unet(num_classes: int = 7, encoder_name: str = "resnet18",
               encoder_weights=None, in_channels: int = 3):
    """构建 U-Net。

    Args:
        num_classes: 语义类别数（LoveDA 为 7）。
        encoder_name: 编码器，默认 resnet18（轻量，适合 CPU）。
        encoder_weights: 预训练权重，None 表示随机初始化（避免下载）。
        in_channels: 输入通道数（RGB 为 3）。
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
