# -*- coding: utf-8 -*-
"""A* 路径规划（8 邻域、禁止穿角、可选视线平滑）。

基线模式只用几何距离；改进模式叠加地形、安全距离、软风险与爬升代价。
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.config import CostWeights, DEFAULT_WEIGHTS
from src.planner.costmap import get_surface
from src.simulator.scene import Scene

NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def _nearest_free(scene: Scene, rc: Tuple[int, int], blocked: np.ndarray, max_radius: int = 30) -> Optional[Tuple[int, int]]:
    r0, c0 = rc
    if scene.in_bounds(r0, c0) and not blocked[r0, c0]:
        return r0, c0
    seen = {(r0, c0)}
    q = deque([(r0, c0, 0)])
    while q:
        r, c, d = q.popleft()
        if d > max_radius:
            break
        if scene.in_bounds(r, c) and not blocked[r, c]:
            return r, c
        for dr, dc, _ in NEIGHBORS[:4]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in seen and scene.in_bounds(nr, nc):
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return None


def _octile(dr: int, dc: int) -> float:
    dr, dc = abs(dr), abs(dc)
    return (dr + dc) + (math.sqrt(2) - 2.0) * min(dr, dc)


def astar(
    scene: Scene,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    mode: str = "improved",
    weights: CostWeights = DEFAULT_WEIGHTS,
    blocked_extra: Optional[np.ndarray] = None,
    max_expand: int = 400_000,
) -> Dict:
    """返回 dict(path=[(x,y)...], length_m, expanded, cost, start_xy, goal_xy)。"""
    blocked = scene.blocked(blocked_extra)
    surf = get_surface(scene, mode, weights)
    sr, sc = scene.xy_to_rc(start_xy)
    gr, gc = scene.xy_to_rc(goal_xy)
    start = _nearest_free(scene, (sr, sc), blocked)
    goal = _nearest_free(scene, (gr, gc), blocked)
    if start is None or goal is None:
        raise RuntimeError(f"A*: 起终点附近没有可通行栅格 {start_xy} -> {goal_xy}")

    nx, ny = scene.nx, scene.ny
    N = nx * ny
    gscore = np.full(N, np.inf, dtype=np.float64)
    parent = np.full(N, -1, dtype=np.int64)
    closed = np.zeros(N, dtype=bool)
    s_idx = start[0] * nx + start[1]
    goal_idx = goal[0] * nx + goal[1]
    gscore[s_idx] = 0.0
    h0 = _octile(goal[0] - start[0], goal[1] - start[1]) * scene.cell
    heap: List[Tuple[float, float, int]] = [(h0, 0.0, s_idx)]
    expanded = 0
    elev = scene.elevation
    cell = scene.cell
    w_climb = weights.w_climb if mode == "improved" else 0.0

    while heap:
        f, g, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        if idx == goal_idx:
            break
        closed[idx] = True
        expanded += 1
        if expanded > max_expand:
            break
        r, c = divmod(idx, nx)
        gr_, gc_ = divmod(goal_idx, nx)
        for dr, dc, unit in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= ny or nc < 0 or nc >= nx:
                continue
            nidx = nr * nx + nc
            if blocked[nr, nc] or closed[nidx]:
                continue
            if dr != 0 and dc != 0:
                if blocked[r + dr, c] or blocked[r, c + dc]:
                    continue
            step = unit * cell
            cost = step * float(surf[nr, nc])
            if w_climb:
                dz = float(elev[nr, nc] - elev[r, c])
                if dz > 0:
                    cost += w_climb * dz
            ng = g + cost
            if ng + 1e-9 < gscore[nidx]:
                gscore[nidx] = ng
                parent[nidx] = idx
                h = _octile(gr_ - nr, gc_ - nc) * cell
                heapq.heappush(heap, (ng + h, ng, nidx))

    if parent[goal_idx] < 0 and goal_idx != s_idx:
        raise RuntimeError(f"A*: 未找到路径 {start_xy} -> {goal_xy}（expanded={expanded}）")

    idx = goal_idx
    rc_path = []
    while idx != -1:
        rc_path.append(divmod(int(idx), nx))
        if idx == s_idx:
            break
        idx = int(parent[idx])
    rc_path.reverse()
    xy_path = [scene.rc_to_xy(r, c) for r, c in rc_path]
    xy_path = simplify_path(scene, xy_path, mode, weights, blocked_extra)
    length = float(np.linalg.norm(np.diff(np.asarray(xy_path), axis=0), axis=1).sum()) if len(xy_path) > 1 else 0.0
    return {
        "path": [(float(p[0]), float(p[1])) for p in xy_path],
        "length_m": length,
        "expanded": int(expanded),
        "cost": float(gscore[goal_idx]),
        "start_xy": tuple(float(v) for v in start_xy),
        "goal_xy": tuple(float(v) for v in goal_xy),
    }


def _segment_clear(
    scene: Scene,
    p0: Sequence[float],
    p1: Sequence[float],
    mode: str,
    weights: CostWeights,
    blocked_extra: Optional[np.ndarray],
) -> bool:
    blocked = scene.blocked(blocked_extra)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(d / (scene.cell * 0.1))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    clear_need = 2.5 if mode == "improved" else 0.0
    for t in ts:
        p = p0 * (1 - t) + p1 * t
        r, c = scene.nearest_index(p)
        if blocked[r, c]:
            return False
        if clear_need and scene.dist_hard[r, c] < clear_need:
            return False
    return True


def simplify_path(
    scene: Scene,
    path_xy: List[np.ndarray],
    mode: str = "improved",
    weights: CostWeights = DEFAULT_WEIGHTS,
    blocked_extra: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """视线法抽稀：在保证不穿越障碍的前提下尽量保留长直线段。"""
    if len(path_xy) <= 2:
        return path_xy
    out: List[np.ndarray] = [np.asarray(path_xy[0], dtype=float)]
    i = 0
    n = len(path_xy)
    while i < n - 1:
        j = n - 1
        while j > i + 1:
            if _segment_clear(scene, path_xy[i], path_xy[j], mode, weights, blocked_extra):
                break
            j -= 1
        out.append(np.asarray(path_xy[j], dtype=float))
        i = j
    return out


def path_hits_mask(scene: Scene, path_xy: Sequence[Sequence[float]], mask: np.ndarray, step: float = 2.5) -> float:
    """路径落在 mask=True 区域内的长度（m）。"""
    if len(path_xy) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path_xy[:-1], path_xy[1:]):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        d = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(d / step)))
        for k in range(n):
            p = a + (b - a) * (k + 0.5) / n
            r, c = scene.nearest_index(p)
            if mask[r, c]:
                total += d / n
    return total


def path_clearance_stats(scene: Scene, path_xy: Sequence[Sequence[float]]) -> Dict:
    """统计路径的安全指标：最小/平均障碍距离、违规长度、穿越障碍次数。"""
    dmin = 1e9
    dsum = 0.0
    dn = 0
    viol_len = 0.0
    near_len = 0.0
    hits = 0
    for a, b in zip(path_xy[:-1], path_xy[1:]):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        d = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(d / 1.0)))
        for k in range(n):
            p = a + (b - a) * (k + 0.5) / n
            r, c = scene.nearest_index(p)
            dist = float(scene.dist_hard[r, c])
            dmin = min(dmin, dist)
            dsum += dist
            dn += 1
            if dist < 3.0:
                viol_len += d / n
            if dist < 8.0:
                near_len += d / n
            if scene.hard[r, c]:
                hits += 1
    return {
        "min_clearance_m": 0.0 if dmin > 1e8 else dmin,
        "mean_clearance_m": 0.0 if dn == 0 else dsum / dn,
        "violation_length_m": viol_len,
        "near_obstacle_length_m": near_len,
        "obstacle_hits": hits,
        "n_samples": dn,
    }

