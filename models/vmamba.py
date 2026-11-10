"""VMamba 分割网络 —— 结构参考实现（需 GPU）。

⚠️ 本机为 CPU-only，`mamba-ssm` 依赖 CUDA 编译（causal-conv1d + 选择性扫描
CUDA 内核），无法安装/运行，故此模型**不能在本机训练或推理**。此文件仅提供
结构代码，供在有 GPU 的机器上使用。

在 GPU 机器上：
    pip install mamba-ssm causal-conv1d
然后 `build_vmamba(num_classes=7)` 即可接入现有训练/评估循环。

参考资产（本机已有）：`D:\\Mamba-UNet`（含 SegMamba / Vim / VMamba 结构与
训练权重 Dice≈0.89），以及其 `mamba/mamba_ssm` 源码。
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except ImportError:  # CPU-only 或未安装
    Mamba = None
    _HAS_MAMBA = False


class VSSBlock(nn.Module):
    """视觉状态空间块：卷积归一化 + 双向 Mamba 扫描 + 残差。

    将 2D 特征图按双向展平为序列，用 SSM 建模长程依赖，再还原为 2D。
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        xf = self.norm(x.flatten(2).transpose(1, 2))          # (B, H*W, C)
        fwd = self.mamba_fwd(xf)
        bwd = self.mamba_bwd(xf.flip(1)).flip(1)
        out = (fwd + bwd).transpose(1, 2).reshape(B, C, H, W)
        return x + out


class Down(nn.Module):
    """下采样：Patch 合并 + VSS 块。"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)
        self.block = VSSBlock(dim_out)

    def forward(self, x):
        return self.block(self.proj(x))


class Up(nn.Module):
    """上采样 + 跳跃连接 + VSS 块。"""

    def __init__(self, dim_in: int, dim_skip: int, dim_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.block = VSSBlock(dim_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = x + skip
        return self.block(x)


class VMambaSeg(nn.Module):
    """VMamba U-Net 风格分割网络（参考实现，需 GPU + mamba-ssm）。"""

    def __init__(self, num_classes: int = 7, in_channels: int = 3, dims=(64, 128, 256, 512)):
        super().__init__()
        if not _HAS_MAMBA:
            raise RuntimeError(
                "mamba-ssm 未安装（CPU-only 环境无法安装）。请在 GPU 机器上 "
                "`pip install mamba-ssm causal-conv1d` 后使用。")
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)
        self.enc = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.enc.append(Down(dims[i], dims[i + 1]))
        self.bottleneck = VSSBlock(dims[-1])
        self.dec = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.dec.append(Up(dims[i], dims[i - 1], dims[i - 1]))
        self.head = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for down in self.enc:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        skips = skips[:-1][::-1]
        for i, up in enumerate(self.dec):
            x = up(x, skips[i])
        return self.head(x)


def build_vmamba(num_classes: int = 7, in_channels: int = 3, **kwargs):
    """构建 VMamba 分割网络（需 GPU + mamba-ssm）。"""
    return VMambaSeg(num_classes=num_classes, in_channels=in_channels)
