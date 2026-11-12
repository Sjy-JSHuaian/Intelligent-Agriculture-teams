# -*- coding: utf-8 -*-
"""轻量几何工具：不依赖 shapely / geopandas，只使用 numpy。"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Polygon = np.ndarray


def polygon_area(poly: np.ndarray) -> float:
    """鞋带公式，返回正面积。"""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return p.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
    return np.array([cx, cy])


def point_in_polygon(pt: Sequence[float], poly: np.ndarray) -> bool:
    """射线法。"""
    x, y = float(pt[0]), float(pt[1])
    p = np.asarray(poly, dtype=float)
    inside = False
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def polygon_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    p = np.asarray(poly, dtype=float)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def rotate(points: np.ndarray, angle_rad: float, origin: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    p = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    return np.column_stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c]) + np.asarray(origin)


def rect_polygon(center: Sequence[float], length: float, width: float, angle_rad: float) -> np.ndarray:
    """以 center 为中心、长轴沿 angle 的矩形。"""
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=float)
    return rotate(local, angle_rad) + np.asarray(center, dtype=float)


def pca_angle(poly: np.ndarray) -> float:
    """多边形主方向（最长轴），返回弧度，范围 [-pi/2, pi/2)。"""
    p = np.asarray(poly, dtype=float)
    q = p - p.mean(axis=0)
    cov = np.cov(q.T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    ang = math.atan2(v[1], v[0])
    if ang >= math.pi / 2:
        ang -= math.pi
    if ang < -math.pi / 2:
        ang += math.pi
    return ang


def resample_segment(p0: Sequence[float], p1: Sequence[float], step: float) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(d / max(step, 1e-6))) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return p0[None, :] * (1 - t) + p1[None, :] * t


def polyline_length(points: Iterable[Sequence[float]]) -> float:
    p = np.asarray(list(points), dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain，返回逆时针顶点。"""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, dtype=float), 6))))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def line_clip_polygon(p0: np.ndarray, p1: np.ndarray, poly: np.ndarray, n: int = 240) -> List[np.ndarray]:
    """将线段按采样点裁到多边形内部，返回若干连续区间端点。"""
    ts = np.linspace(0.0, 1.0, n)
    pts = p0[None, :] * (1 - ts)[:, None] + p1[None, :] * ts[:, None]
    inside = np.array([point_in_polygon(pt, poly) for pt in pts])
    intervals: List[np.ndarray] = []
    start = None
    for i, ok in enumerate(inside):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 1:
                intervals.append(np.vstack([pts[start], pts[i - 1]]))
            start = None
    if start is not None:
        intervals.append(np.vstack([pts[start], pts[-1]]))
    return intervals


def distance_point_to_polyline(pt: np.ndarray, line: np.ndarray) -> float:
    """点到折线的最短距离。"""
    if len(line) == 0:
        return float("inf")
    if len(line) == 1:
        return float(np.linalg.norm(pt - line[0]))
    best = float("inf")
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(pt - a, ab) / denom, 0.0, 1.0))
        q = a + t * ab
        best = min(best, float(np.linalg.norm(pt - q)))
    return best
