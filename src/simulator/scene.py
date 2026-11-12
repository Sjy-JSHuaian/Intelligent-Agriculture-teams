# -*- coding: utf-8 -*-
"""场景数据模型与读取。

场景目录固定包含：
    grid.npz        栅格数据（高程/坡度/坡向/粗糙度/障碍/地块ID/障碍距离）
    scene.json      元数据、地块列表、起降点、来源说明
    fields.geojson  地块边界（QGIS / 网页可直接打开）
    boundary.geojson
    obstacles.geojson
    dem.tif         5 m DEM（EPSG:32648，QGIS 底图）
    field_id.tif    地块编号栅格
    *.shp           Shapefile 版本（QGIS 编辑/出图）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.config import CELL_M, SCENE_DIR, SCENE_SPECS, UAVSpec, DEFAULT_UAV


@dataclass
class Field:
    field_id: str
    polygon: np.ndarray
    risk: float
    risk_level: str
    priority: int
    area_m2: float
    centroid: np.ndarray
    mean_elev: float
    row: int = 0
    col: int = 0

    @property
    def risk_tier(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.risk_level]


class Scene:
    """一套梯田仿真场景（局部平面坐标，x 向东、y 向北，单位 m）。"""

    def __init__(
        self,
        scene_id: str,
        meta: Dict,
        elevation: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        roughness: np.ndarray,
        hard: np.ndarray,
        soft: np.ndarray,
        field_id: np.ndarray,
        dist_hard: np.ndarray,
        fields: List[Field],
        boundary: np.ndarray,
        depot: np.ndarray,
        origin_xy: Tuple[float, float],
    ):
        self.scene_id = scene_id
        self.meta = meta
        self.name = meta.get("name", scene_id)
        self.complexity = meta.get("complexity", "unknown")
        self.cell = float(meta.get("cell_m", CELL_M))
        self.nx = int(meta["nx"])
        self.ny = int(meta["ny"])
        self.size_m = float(meta.get("size_m", self.nx * self.cell))
        # 规划使用局部坐标 (0..size)；GeoTIFF 使用 UTM 绝对坐标
        self.utm_origin = np.asarray(meta.get("origin_xy", (0.0, 0.0)), dtype=float)
        self.origin_xy = np.zeros(2, dtype=float)
        self.elevation = elevation.astype(np.float32)
        self.slope = slope.astype(np.float32)
        self.aspect = aspect.astype(np.float32)
        self.roughness = roughness.astype(np.float32)
        self.hard = hard.astype(bool)
        self.soft = soft.astype(np.float32)
        self.field_id = field_id.astype(np.int32)
        self.dist_hard = dist_hard.astype(np.float32)
        self.fields = fields
        self.boundary = boundary
        self.depot = np.asarray(depot, dtype=float)
        self.field_by_id = {f.field_id: f for f in fields}
        self._blocked_cache: Dict[str, np.ndarray] = {}
        self._risk_grid: Optional[np.ndarray] = None

    # ------------------------------------------------------------ 坐标
    def xy_to_rc(self, xy: Sequence[float]) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        c = int(np.floor((x - self.origin_xy[0]) / self.cell))
        r = int(np.floor((self.origin_xy[1] + self.size_m - y) / self.cell))
        return r, c

    def rc_to_xy(self, r: int, c: int) -> np.ndarray:
        x = self.origin_xy[0] + (c + 0.5) * self.cell
        y = self.origin_xy[1] + self.size_m - (r + 0.5) * self.cell
        return np.array([x, y], dtype=float)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.ny and 0 <= c < self.nx

    def contains_xy(self, xy: Sequence[float]) -> bool:
        r, c = self.xy_to_rc(xy)
        return self.in_bounds(r, c)

    def nearest_index(self, xy: Sequence[float]) -> Tuple[int, int]:
        r, c = self.xy_to_rc(xy)
        r = int(np.clip(r, 0, self.ny - 1))
        c = int(np.clip(c, 0, self.nx - 1))
        return r, c

    def sample(self, arr: np.ndarray, xy: Sequence[float]) -> float:
        """双线性采样（用于高程）。"""
        x, y = float(xy[0]), float(xy[1])
        fc = (x - self.origin_xy[0]) / self.cell - 0.5
        fr = (self.origin_xy[1] + self.size_m - y) / self.cell - 0.5
        c0 = int(np.floor(fc)); r0 = int(np.floor(fr))
        tx = fc - c0; ty = fr - r0
        c1 = min(max(c0 + 1, 0), self.nx - 1); r1 = min(max(r0 + 1, 0), self.ny - 1)
        c0 = min(max(c0, 0), self.nx - 1); r0 = min(max(r0, 0), self.ny - 1)
        v = (
            arr[r0, c0] * (1 - tx) * (1 - ty)
            + arr[r0, c1] * tx * (1 - ty)
            + arr[r1, c0] * (1 - tx) * ty
            + arr[r1, c1] * tx * ty
        )
        return float(v)

    def elevation_at(self, xy: Sequence[float]) -> float:
        return self.sample(self.elevation, xy)

    def ground_z(self, xy: Sequence[float]) -> float:
        return self.elevation_at(xy)

    # ------------------------------------------------------------ 地块
    def field_mask(self, fid: str) -> np.ndarray:
        idx = int(fid.replace("T", ""))
        return self.field_id == idx

    def field_cells(self, fid: str) -> np.ndarray:
        return np.argwhere(self.field_mask(fid))

    def risk_grid(self) -> np.ndarray:
        if self._risk_grid is None:
            g = np.zeros((self.ny, self.nx), dtype=np.float32)
            for f in self.fields:
                g[self.field_id == int(f.field_id[1:])] = f.risk
            self._risk_grid = g
        return self._risk_grid

    # ------------------------------------------------------------ 通行性
    def blocked(self, extra: Optional[np.ndarray] = None) -> np.ndarray:
        """不可通行栅格：障碍物 + 过陡坡 + 边界。"""
        key = "extra" if extra is not None else "base"
        if key not in self._blocked_cache:
            b = self.hard.copy()
            b |= self.slope > 38.0
            b[0, :] = b[-1, :] = True
            b[:, 0] = b[:, -1] = True
            self._blocked_cache[key] = b
        base = self._blocked_cache[key]
        if extra is None:
            return base
        return base | extra

    def clearance_violation(self, xy: Sequence[float], threshold: float = 3.0) -> bool:
        r, c = self.nearest_index(xy)
        return bool(self.dist_hard[r, c] < threshold)

    # ------------------------------------------------------------ 持久化
    @classmethod
    def load(cls, scene_id: str, scene_dir: Optional[Path] = None) -> "Scene":
        d = Path(scene_dir) if scene_dir else SCENE_DIR / scene_id
        with open(d / "scene.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        z = np.load(d / "grid.npz")
        fields = [
            Field(
                field_id=f["field_id"],
                polygon=np.asarray(f["polygon"], dtype=float),
                risk=float(f["risk"]),
                risk_level=f["risk_level"],
                priority=int(f["priority"]),
                area_m2=float(f["area_m2"]),
                centroid=np.asarray(f["centroid"], dtype=float),
                mean_elev=float(f["mean_elev"]),
            )
            for f in meta["fields"]
        ]
        return cls(
            scene_id=scene_id,
            meta=meta,
            elevation=z["elevation"],
            slope=z["slope"],
            aspect=z["aspect"],
            roughness=z["roughness"],
            hard=z["hard"],
            soft=z["soft"],
            field_id=z["field_id"],
            dist_hard=z["dist_hard"],
            fields=fields,
            boundary=np.asarray(meta["boundary"], dtype=float),
            depot=np.asarray(meta["depot_xy"], dtype=float),
            origin_xy=(0.0, 0.0),
        )

    def uav(self) -> UAVSpec:
        u = self.meta.get("uav", {})
        args = {k: v for k, v in u.items() if k in UAVSpec.__dataclass_fields__}
        return UAVSpec(**args) if args else DEFAULT_UAV


def scene_summary(scene: Scene) -> Dict:
    return {
        "scene_id": scene.scene_id,
        "name": scene.name,
        "complexity": scene.complexity,
        "cell_m": scene.cell,
        "nx": scene.nx,
        "ny": scene.ny,
        "size_m": scene.size_m,
        "n_fields": len(scene.fields),
        "elev_min": float(scene.elevation.min()),
        "elev_max": float(scene.elevation.max()),
        "relief_m": float(scene.elevation.max() - scene.elevation.min()),
        "mean_slope_deg": float(scene.slope.mean()),
        "n_obstacle_cells": int(scene.hard.sum()),
        "hard_area_pct": float(scene.hard.mean() * 100),
        "depot_xy": scene.depot.tolist(),
        "origin_xy": scene.utm_origin.tolist(),
        "center_lonlat": scene.meta.get("center_lonlat"),
    }


def load_all(scene_ids: Optional[Sequence[str]] = None) -> List[Scene]:
    ids = list(scene_ids) if scene_ids else list(SCENE_SPECS.keys())
    return [Scene.load(sid) for sid in ids]
