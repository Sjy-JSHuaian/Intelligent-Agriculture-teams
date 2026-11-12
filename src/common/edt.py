# -*- coding: utf-8 -*-
"""栅格辅助：精确欧氏距离变换（不依赖 scipy）与平滑噪声。"""
from __future__ import annotations

import numpy as np


def _edt_1d(f: np.ndarray) -> np.ndarray:
    """一维平方距离变换（Felzenszwalb & Huttenlocher）。"""
    n = f.size
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def distance_transform_edt(mask: np.ndarray, cell: float = 1.0) -> np.ndarray:
    """到最近 True 像元的欧氏距离（米）。mask 为 True 的位置距离为 0。"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.full(m.shape, 1e6, dtype=np.float32)
    f = np.where(m, 0.0, 1e12)
    h, w = f.shape
    for i in range(h):
        f[i, :] = _edt_1d(f[i, :])
    for j in range(w):
        f[:, j] = _edt_1d(f[:, j])
    return (np.sqrt(f) * cell).astype(np.float32)


def box_blur(a: np.ndarray, radius: int = 1) -> np.ndarray:
    """用积分图做一次盒式滤波（边界复制）。"""
    if radius <= 0:
        return a.copy()
    pad = np.pad(a, radius, mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    k = 2 * radius + 1
    out = (
        c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    ) / float(k * k)
    return out[: a.shape[0], : a.shape[1]]


def smooth_noise(nx: int, ny: int, scale_px: int, seed: int, octaves: int = 3) -> np.ndarray:
    """生成 0~1 的平滑空间噪声，用于模拟病害风险场。"""
    rng = np.random.default_rng(seed)
    total = np.zeros((ny, nx), dtype=np.float64)
    amp = 1.0
    norm = 0.0
    scale = max(1, int(scale_px))
    for _ in range(octaves):
        h = max(1, int(np.ceil(ny / scale)))
        w = max(1, int(np.ceil(nx / scale)))
        small = rng.random((h, w))
        big = np.kron(small, np.ones((scale, scale), dtype=float))[:ny, :nx]
        if big.shape != (ny, nx):
            big = np.pad(big, ((0, max(0, ny - big.shape[0])), (0, max(0, nx - big.shape[1]))), mode="edge")[:ny, :nx]
        big = box_blur(box_blur(big, 2), 2)
        total += amp * big
        norm += amp
        amp *= 0.5
        scale = max(1, scale // 2)
    total /= max(norm, 1e-9)
    lo, hi = float(total.min()), float(total.max())
    return (total - lo) / max(hi - lo, 1e-9)
