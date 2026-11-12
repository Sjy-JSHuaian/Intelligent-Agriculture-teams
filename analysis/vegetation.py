"""植被/生态分析（RGB 植被指数，非 NDVI）。

当前影像为 RGB 三通道、无 NIR，故采用 VARI / ExG / GLI / NDI 作为植被覆盖的
近似特征；报告中须注明「RGB 植被指数」而非 NDVI。阈值均为经验值，待真实
加榜梯田影像到位后需校准。
"""
from __future__ import annotations

import numpy as np

from datasets.preprocessing import compute_rgb_indices, to_float01

# 植被判定阈值（经验值）
GLI_VEG_THRESHOLD = 0.0     # GLI > 0 视为植被（较宽松）
VARI_VEG_THRESHOLD = 0.1    # VARI > 0.1 视为植被（较严格）


def vegetation_mask(rgb: np.ndarray, method: str = "GLI",
                    threshold: float | None = None) -> np.ndarray:
    """计算二值植被掩膜。

    Args:
        rgb: HWC uint8 RGB 影像。
        method: "GLI" 或 "VARI"。
        threshold: 覆盖默认阈值。
    返回 bool 数组 (H, W)。
    """
    x = to_float01(rgb)
    indices = compute_rgb_indices(x)
    key = method.upper()
    th = threshold if threshold is not None else (
        GLI_VEG_THRESHOLD if key == "GLI" else VARI_VEG_THRESHOLD)
    return indices[key] > th


def vegetation_fraction(rgb: np.ndarray, method: str = "GLI",
                        threshold: float | None = None) -> float:
    """植被覆盖度 = 植被像素占比。"""
    return float(vegetation_mask(rgb, method, threshold).mean())


def index_statistics(indices: dict[str, np.ndarray]) -> dict:
    """各植被指数的 mean/std/min/max 统计。"""
    return {name: {"mean": float(v.mean()), "std": float(v.std()),
                   "min": float(v.min()), "max": float(v.max())}
            for name, v in indices.items()}


def analyze_vegetation(rgb: np.ndarray) -> dict:
    """完整植被分析：指数图、统计、覆盖度、植被掩膜。

    Args:
        rgb: HWC uint8 RGB 影像。
    """
    x = to_float01(rgb)
    indices = compute_rgb_indices(x)
    return {
        "indices": indices,
        "index_statistics": index_statistics(indices),
        "vegetation_fraction": vegetation_fraction(rgb),
        "vegetation_mask": vegetation_mask(rgb),
    }
