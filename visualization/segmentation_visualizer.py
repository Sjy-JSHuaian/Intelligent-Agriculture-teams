"""分割结果可视化：类别配色、掩膜叠加图。

配色基于原始掩膜取值 0-7（与 dataset_loader.RAW_CLASSES 对应）。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# 可视化配色（非官方标准，仅用于展示；取值与名称见 dataset_loader.RAW_CLASSES）
LOVEDA_COLORS = {
    0: (0, 0, 0),          # no-data     黑
    1: (128, 128, 128),    # background  灰
    2: (255, 0, 0),        # building    红
    3: (255, 255, 0),      # road        黄
    4: (0, 0, 255),        # water       蓝
    5: (165, 42, 42),      # barren      棕
    6: (0, 128, 0),        # forest      深绿
    7: (128, 255, 0),      # agriculture 黄绿
}


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """单通道类别掩膜 -> RGB 彩色图 (H, W, 3) uint8。"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in LOVEDA_COLORS.items():
        color[mask == cid] = rgb
    return color


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将掩膜叠加到影像上，返回 RGB uint8。"""
    color_mask = mask_to_color(mask)
    return cv2.addWeighted(image.astype(np.uint8), 1 - alpha, color_mask, alpha, 0)


def _imwrite(out_path: str, img_bgr: np.ndarray) -> None:
    """写图（中文路径安全：imencode + tofile）。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise IOError(f"无法编码图像: {out_path}")
    buf.tofile(str(out_path))


def save_overlay(image: np.ndarray, mask: np.ndarray, out_path: str, alpha: float = 0.5) -> str:
    """保存叠加图，返回输出路径。"""
    out = overlay(image, mask, alpha)
    _imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return str(out_path)


def save_color_mask(mask: np.ndarray, out_path: str) -> str:
    """保存彩色掩膜，返回输出路径。"""
    color = mask_to_color(mask)
    _imwrite(str(out_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    return str(out_path)
