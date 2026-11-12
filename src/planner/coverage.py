# -*- coding: utf-8 -*-
"""地块覆盖航带生成（牛耕式），并提供 4 种入口/出口变体供路径优化选择。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.common.config import DEFAULT_UAV, UAVSpec
from src.common.geometry import line_clip_polygon, pca_angle, polyline_length
from src.simulator.scene import Field, Scene


@dataclass
class CoveragePlan:
    waypoints: List[np.ndarray]
    entry: np.ndarray
    exit: np.ndarray
    length_m: float
    n_strips: int
    variant: int
    heading_deg: float


def _build(
    field: Field,
    uav: UAVSpec,
    overlap: float,
    cross: bool,
    reverse: bool,
    variant: int,
) -> CoveragePlan:
    poly = np.asarray(field.polygon, dtype=float)
    center = poly.mean(axis=0)
    angle = pca_angle(poly) + (math.pi / 2.0 if cross else 0.0)
    c, s = math.cos(angle), math.sin(angle)
    # 世界 -> (u 沿航向, v 横向)
    q = poly - center
    uv = np.column_stack([q[:, 0] * c + q[:, 1] * s, -q[:, 0] * s + q[:, 1] * c])
    umin, umax = float(uv[:, 0].min()), float(uv[:, 0].max())
    vmin, vmax = float(uv[:, 1].min()), float(uv[:, 1].max())
    width = max(vmax - vmin, 1e-6)
    swath = uav.swath(overlap)
    n_strips = max(1, int(math.ceil(width / swath - 1e-9)))
    if n_strips == 1:
        offsets = np.array([(vmin + vmax) / 2.0])
    else:
        offsets = np.linspace(vmin + swath / 2.0, vmax - swath / 2.0, n_strips)
    step = uav.capture_step(overlap)
    pts: List[np.ndarray] = []
    for k, v in enumerate(offsets):
        p0 = np.array([umin - 2.0, v]); p1 = np.array([umax + 2.0, v])
        intervals = line_clip_polygon(p0, p1, uv, n=360)
        if not intervals:
            continue
        seg = max(intervals, key=lambda s: abs(s[1, 0] - s[0, 0]))
        u0, u1 = float(min(seg[0, 0], seg[1, 0])), float(max(seg[0, 0], seg[1, 0]))
        if u1 - u0 < 1.0:
            continue
        us = list(np.arange(u0, u1, step))
        if not us or abs(us[-1] - u1) > 1e-6:
            us.append(u1)
        if len(us) < 2:
            us = [u0, u1]
        if k % 2 == 1:
            us = us[::-1]
        for u in us:
            pts.append(np.array([u, v]))
    if reverse:
        pts = pts[::-1]
    # 局部 (u,v) -> 世界
    world = []
    for u, v in pts:
        x = center[0] + u * c - v * s
        y = center[1] + u * s + v * c
        world.append(np.array([x, y]))
    if len(world) < 2:
        world = [center.copy(), center.copy() + np.array([0.1, 0.0])]
    length = polyline_length(world)
    heading = math.degrees(math.atan2(world[-1][1] - world[0][1], world[-1][0] - world[0][0]))
    return CoveragePlan(world, world[0], world[-1], length, n_strips, variant, heading)


def variants(field: Field, uav: UAVSpec = DEFAULT_UAV, overlap: float = 0.15) -> List[CoveragePlan]:
    """4 个变体：沿主方向/垂直方向 × 正/反飞行。"""
    plans = []
    for vi, (cross, rev) in enumerate([(False, False), (False, True), (True, False), (True, True)]):
        plans.append(_build(field, uav, overlap, cross, rev, vi))
    return plans


def choose_variant(
    plans: Sequence[CoveragePlan],
    prev_xy: Sequence[float],
    next_xy: Optional[Sequence[float]] = None,
) -> CoveragePlan:
    """选择入口/出口使“上一段进场 + 本块航带 + 离场”最短（mUDAI 视角优化的落点）。"""
    best = None
    best_cost = float("inf")
    prev = np.asarray(prev_xy, dtype=float)
    nxt = np.asarray(next_xy, dtype=float) if next_xy is not None else None
    for p in plans:
        cost = float(np.linalg.norm(p.entry - prev))
        if nxt is not None:
            cost += float(np.linalg.norm(p.exit - nxt))
        if cost < best_cost:
            best, best_cost = p, cost
    return best if best is not None else plans[0]
