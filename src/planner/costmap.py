# -*- coding: utf-8 -*-
"""代价地图：基线与改进 A* 的栅格通行代价。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.common.config import CostWeights, DEFAULT_WEIGHTS
from src.simulator.scene import Scene


def cost_surface(scene: Scene, mode: str, weights: CostWeights = DEFAULT_WEIGHTS) -> np.ndarray:
    """返回每个栅格的通行代价倍率（不含方向相关的爬升项）。

    baseline：恒为 1，仅用几何距离。
    improved：坡度平方惩罚 + 障碍安全距离惩罚 + 道路/沟渠软风险。
    """
    if mode == "baseline":
        return np.ones((scene.ny, scene.nx), dtype=np.float32)

    slope_term = (scene.slope / weights.slope_ref_deg) ** 2
    clear = np.clip((weights.d_safe - scene.dist_hard) / weights.d_safe, 0.0, 1.0) ** 1.5
    soft = np.clip(scene.soft, 0.0, 1.0)
    surf = 1.0 + weights.w_slope * slope_term + weights.w_clear * clear + weights.w_soft * soft
    return surf.astype(np.float32)


def get_surface(scene: Scene, mode: str, weights: CostWeights = DEFAULT_WEIGHTS) -> np.ndarray:
    cache = getattr(scene, "_surface_cache", None)
    if cache is None:
        cache = {}
        scene._surface_cache = cache  # type: ignore[attr-defined]
    key = (mode, id(weights))
    if key not in cache:
        cache[key] = cost_surface(scene, mode, weights)
    return cache[key]
