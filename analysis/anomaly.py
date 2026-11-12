"""异常检测：植被胁迫（低绿度斑块）、裸地、地块变化。

均为基于 RGB 植被指数的启发式方法（无标签监督），阈值经验化，
结果作为异常候选区域，供后续人工复核或结合时序数据验证。
"""
from __future__ import annotations

import numpy as np

from datasets.preprocessing import compute_exg, compute_gli, compute_vari, to_float01


def detect_low_vigor(rgb: np.ndarray, z_threshold: float = 1.0,
                     veg_threshold: float = 0.0) -> np.ndarray:
    """植被胁迫/低绿度异常：植被区域内 GLI 显著低于植被均值的像素。

    做法：先以 GLI > veg_threshold 圈定植被，再在该区域内按 z-score
    找出 GLI 显著偏低（z < -z_threshold）的斑块（疑似枯黄/病虫害/缺水）。
    返回 bool 掩膜 (H, W)。
    """
    x = to_float01(rgb)
    gli = compute_gli(x)
    veg = gli > veg_threshold
    anomaly = np.zeros_like(gli, dtype=bool)
    if veg.any():
        mu, sd = float(gli[veg].mean()), float(gli[veg].std())
        if sd > 1e-6:
            z = (gli - mu) / sd
            anomaly = veg & (z < -z_threshold)
    return anomaly


def detect_bare_land(rgb: np.ndarray, exg_threshold: float = -0.05,
                     min_brightness: float = 0.1,
                     max_brightness: float = 0.9) -> np.ndarray:
    """裸地检测：ExG 偏低且呈土壤色调（红/棕分量占优）。

    排除过暗（阴影）与过亮（高反光）像素。返回 bool 掩膜 (H, W)。
    """
    x = to_float01(rgb)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    exg = compute_exg(x)
    brightness = x.mean(axis=-1)
    soil = (exg < exg_threshold) & (r > g) & (r > b)
    soil &= (brightness > min_brightness) & (brightness < max_brightness)
    return soil


def detect_change(rgb_a: np.ndarray, rgb_b: np.ndarray,
                  index: str = "VARI", threshold: float = 0.15) -> np.ndarray:
    """地块变化检测：两期影像植被指数差值的绝对值超过阈值视为变化。

    Args:
        rgb_a, rgb_b: 同一位置两个时相的 HWC uint8 RGB 影像（需已配准、同尺寸）。
        index: "VARI" 或 "GLI"。
        threshold: 差值阈值。
    返回 bool 掩膜 (H, W)。
    """
    xa, xb = to_float01(rgb_a), to_float01(rgb_b)
    if xa.shape != xb.shape:
        raise ValueError(f"两期影像尺寸不一致: {xa.shape} vs {xb.shape}")
    if index.upper() == "VARI":
        ia, ib = compute_vari(xa), compute_vari(xb)
    else:
        ia, ib = compute_gli(xa), compute_gli(xb)
    return np.abs(ia - ib) > threshold
