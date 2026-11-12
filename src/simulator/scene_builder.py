# -*- coding: utf-8 -*-
"""从真实 DEM + OSM 构建 3 套 5 m 梯田仿真场景。

真实数据：Copernicus DEM GLO-30（地形）、OpenStreetMap（道路/水系/林地等）。
仿真数据：地块边界、田埂、建筑、树木、电线等按固定随机种子人工布设，
          并在 scene.json 的 sources/notes 中逐项标注，绝不冒充实测。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform as warp_transform

from src.common.config import CELL_M, CRS_UTM, DEFAULT_UAV, SCENE_SPECS, SceneSpec
from src.common.edt import box_blur, distance_transform_edt
from src.common.geometry import (
    point_in_polygon, polygon_area, polygon_centroid, rect_polygon, distance_point_to_polyline,
)

LOCAL_TRANSFORM_CACHE: Dict[str, object] = {}


# ---------------------------------------------------------------- 基础工具
def _geom_poly(poly: np.ndarray) -> Dict:
    ring = [(float(x), float(y)) for x, y in poly]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _raster_poly(poly: np.ndarray, nx: int, ny: int, transform) -> np.ndarray:
    return rasterize([(_geom_poly(poly), 1)], out_shape=(ny, nx), transform=transform,
                     fill=0, dtype="uint8", all_touched=False).astype(bool)


def _raster_polys(polys: Sequence[np.ndarray], nx: int, ny: int, transform) -> np.ndarray:
    if not polys:
        return np.zeros((ny, nx), dtype=bool)
    shapes = [(_geom_poly(p), 1) for p in polys]
    return rasterize(shapes, out_shape=(ny, nx), transform=transform, fill=0,
                     dtype="uint8", all_touched=False).astype(bool)


def _dilate(mask: np.ndarray, radius_m: float, cell: float) -> np.ndarray:
    if radius_m <= 0:
        return mask.copy()
    return distance_transform_edt(mask, cell) <= radius_m


def _line_mask(lines: Sequence[Sequence[Tuple[float, float]]], nx: int, ny: int, size: float, cell: float, radius_m: float) -> np.ndarray:
    m = np.zeros((ny, nx), dtype=bool)
    if not lines:
        return m
    for line in lines:
        for a, b in zip(line[:-1], line[1:]):
            a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
            d = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(d / (cell * 0.7))))
            for k in range(n + 1):
                p = a + (b - a) * k / n
                c = int(np.floor(p[0] / cell)); r = int(np.floor((size - p[1]) / cell))
                if 0 <= r < ny and 0 <= c < nx:
                    m[r, c] = True
    return _dilate(m, radius_m, cell)


def _mask_to_geojson_polygons(mask: np.ndarray, transform) -> List[np.ndarray]:
    from rasterio.features import shapes as _shapes
    out = []
    for geom, val in _shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if int(val) == 1 and geom["type"] == "Polygon":
            out.append(np.asarray(geom["coordinates"][0], dtype=float))
    return out


def _clip_segment(a: Tuple[float, float], b: Tuple[float, float], lo: float, hi: float):
    """Liang-Barsky 线段裁剪到 [lo,hi]×[lo,hi]。"""
    x0, y0 = float(a[0]), float(a[1])
    x1, y1 = float(b[0]), float(b[1])
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - lo, hi - x0, y0 - lo, hi - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    if u1 > u2:
        return None
    return (x0 + u1 * dx, y0 + u1 * dy), (x0 + u2 * dx, y0 + u2 * dy)


def _clip_line(line: np.ndarray, lo: float, hi: float) -> List[np.ndarray]:
    """折线逐段裁剪，返回若干条落在场景内的折线。"""
    pts = np.asarray(line, dtype=float)
    parts: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = _clip_segment(a, b, lo, hi)
        if seg is None:
            if len(cur) >= 2:
                parts.append(cur)
            cur = []
            continue
        s0, s1 = seg
        if cur and abs(cur[-1][0] - s0[0]) < 1e-6 and abs(cur[-1][1] - s0[1]) < 1e-6:
            cur.append(s1)
        else:
            if len(cur) >= 2:
                parts.append(cur)
            cur = [s0, s1]
    if len(cur) >= 2:
        parts.append(cur)
    return [np.asarray(p, dtype=float) for p in parts]


# ---------------------------------------------------------------- OSM
def load_osm(osm_path: Optional[Path], x0: float, y0: float, size: float, margin: float = 200.0) -> Dict:
    empty = {"roads": [], "tracks": [], "paths": [], "waterways": [], "farmland": [], "forest": [], "buildings": [], "power": []}
    if not osm_path or not Path(osm_path).exists():
        return empty
    with open(osm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = {k: [] for k in empty}
    xmin, ymin, xmax, ymax = -margin, -margin, size + margin, size + margin
    for e in data.get("elements", []):
        tags = e.get("tags", {}) or {}
        geom = e.get("geometry")
        if not geom:
            continue
        lons = [q["lon"] for q in geom]; lats = [q["lat"] for q in geom]
        xs, ys = warp_transform("EPSG:4326", CRS_UTM, lons, lats)
        pts = np.column_stack([np.asarray(xs) - x0, np.asarray(ys) - y0])
        if not ((pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)).any():
            continue
        if "highway" in tags:
            hw = tags["highway"]
            if hw in ("footway", "path", "steps", "cycleway", "bridleway"):
                res["paths"].append(pts)
            elif hw in ("track",):
                res["tracks"].append(pts)
            else:
                res["roads"].append(pts)
        if "waterway" in tags:
            res["waterways"].append(pts)
        if "building" in tags and len(pts) >= 3:
            res["buildings"].append(pts)
        if tags.get("landuse") in ("farmland", "orchard") and len(pts) >= 3:
            res["farmland"].append(pts)
        if tags.get("natural") in ("wood", "scrub") and len(pts) >= 3:
            res["forest"].append(pts)
        if "power" in tags and tags.get("power") in ("line", "minor_line", "cable"):
            res["power"].append(pts)
    return res


# ---------------------------------------------------------------- 场景构建
class SceneBuilder:
    def __init__(self, spec: SceneSpec, dem_path: Path, osm_path: Optional[Path], out_dir: Path):
        self.spec = spec
        self.dem_path = Path(dem_path)
        self.osm_path = osm_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(spec.seed)
        self.cell = CELL_M
        self.size = float(spec.size_m)
        self.nx = int(round(self.size / self.cell))
        self.ny = self.nx
        self.transform_local = from_origin(0.0, self.size, self.cell, self.cell)
        self.transform_abs = None
        self.origin = None

    # -------------------------------------------------- 地形
    def prepare_terrain(self):
        lon, lat = self.spec.center_lon, self.spec.center_lat
        cx, cy = warp_transform("EPSG:4326", CRS_UTM, [lon], [lat])
        cx, cy = float(cx[0]), float(cy[0])
        self.origin = (cx - self.size / 2.0, cy - self.size / 2.0)
        self.transform_abs = from_origin(self.origin[0], self.origin[1] + self.size, self.cell, self.cell)
        dst = np.zeros((self.ny, self.nx), dtype=np.float32)
        with rasterio.open(self.dem_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=self.transform_abs, dst_crs=CRS_UTM,
                resampling=Resampling.bilinear,
            )
        self.elevation = dst
        gy, gx = np.gradient(dst, self.cell, self.cell)
        self.gx, self.gy = gx, gy
        self.slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
        self.aspect = (np.degrees(np.arctan2(-gx, gy)) % 360.0).astype(np.float32)
        mean = box_blur(box_blur(dst.astype(np.float64), 1), 1)
        mean2 = box_blur(box_blur((dst.astype(np.float64)) ** 2, 1), 1)
        self.roughness = np.sqrt(np.maximum(mean2 - mean ** 2, 0)).astype(np.float32)

    def grad_angle(self, xy: Sequence[float]) -> float:
        """局部等高线方向（弧度）。"""
        x, y = float(xy[0]), float(xy[1])
        gx = self._sample_grad(self.gx, x, y)
        gy = self._sample_grad(self.gy, x, y)
        if abs(gx) < 1e-9 and abs(gy) < 1e-9:
            return 0.0
        return math.atan2(gy, gx) + math.pi / 2.0

    def _sample_grad(self, arr: np.ndarray, x: float, y: float) -> float:
        c = int(np.clip((x / self.cell) - 0.5, 0, self.nx - 1))
        r = int(np.clip((self.size - y) / self.cell - 0.5, 0, self.ny - 1))
        return float(arr[r, c])

    def sample_slope(self, xy: Sequence[float]) -> float:
        x, y = float(xy[0]), float(xy[1])
        c = int(np.clip((x / self.cell) - 0.5, 0, self.nx - 1))
        r = int(np.clip((self.size - y) / self.cell - 0.5, 0, self.ny - 1))
        return float(self.slope[r, c])

    # -------------------------------------------------- OSM 图层
    def prepare_osm(self):
        self.osm = load_osm(self.osm_path, self.origin[0], self.origin[1], self.size)
        nx, ny, cell, size = self.nx, self.ny, self.cell, self.size
        self.road_mask = _line_mask(self.osm["roads"], nx, ny, size, cell, 3.0)
        self.track_mask = _line_mask(self.osm["tracks"], nx, ny, size, cell, 2.0)
        self.water_mask = _line_mask(self.osm["waterways"], nx, ny, size, cell, 2.5)
        self.forest_mask = _raster_polys(self.osm["forest"], nx, ny, self.transform_local)
        self.osm_building_polys = list(self.osm["buildings"])
        self.osm_building_mask = _raster_polys(self.osm_building_polys, nx, ny, self.transform_local)
        self.osm_building_mask = _dilate(self.osm_building_mask, 2.5, cell)
        self.dist_road = distance_transform_edt(self.road_mask | self.track_mask, cell)

    # -------------------------------------------------- 建筑
    def place_buildings(self):
        n = int(round(6 + 14 * self.spec.obstacle_level))
        polys: List[np.ndarray] = []
        centers: List[np.ndarray] = []
        # 找村庄中心：靠近道路、坡缓、位于场景内
        candidates = []
        for _ in range(400):
            x = self.rng.uniform(80, self.size - 80); y = self.rng.uniform(80, self.size - 80)
            if self.sample_slope((x, y)) < 18 and self._inside((x, y), 60):
                score = float(self.dist_road[int((self.size - y) / self.cell), int(x / self.cell)]) if self.road_mask.any() else 100.0
                candidates.append((score, x, y))
        if not candidates:
            candidates = [(0, self.size / 2, self.size / 2)]
        candidates.sort(key=lambda t: t[0])
        vx, vy = candidates[0][1], candidates[0][2]
        attempts = 0
        while len(polys) < n and attempts < 1200:
            attempts += 1
            ang = self.rng.uniform(0, math.pi)
            rad = self.rng.uniform(0, 95)
            theta = self.rng.uniform(0, 2 * math.pi)
            c = np.array([vx + rad * math.cos(theta), vy + rad * math.sin(theta)])
            w = self.rng.uniform(6.0, 12.0); h = self.rng.uniform(5.0, 9.0)
            poly = rect_polygon(c, max(w, h), min(w, h), ang)
            if not all(self._inside(p, 12) for p in poly):
                continue
            if self._poly_hits(poly, self.road_mask, 1.5) or self.sample_slope(c) > 22:
                continue
            if any(polygon_area(poly) > 0 and self._polys_overlap(poly, q) for q in polys):
                continue
            polys.append(poly); centers.append(c)
        self.buildings = polys
        self.building_centers = centers
        self.building_mask = _raster_polys(polys, self.nx, self.ny, self.transform_local)
        self.building_mask |= self.osm_building_mask
        self.building_mask = _dilate(self.building_mask, 2.5, self.cell)

    def _poly_hits(self, poly: np.ndarray, mask: np.ndarray, min_dist_m: float) -> bool:
        d = distance_transform_edt(mask, self.cell) if mask.any() else None
        if d is None:
            return False
        for p in poly:
            r, c = self._rc(p)
            if d[r, c] < min_dist_m:
                return True
        return False

    def _polys_overlap(self, a: np.ndarray, b: np.ndarray) -> bool:
        for p in a:
            if point_in_polygon(p, b):
                return True
        for p in b:
            if point_in_polygon(p, a):
                return True
        return point_in_polygon(polygon_centroid(a), b) or point_in_polygon(polygon_centroid(b), a)

    # -------------------------------------------------- 地块
    def place_fields(self):
        n = self.spec.n_fields
        fields = []
        margin = 35.0
        candidates = []
        step = 22.0
        xs = np.arange(margin, self.size - margin, step)
        ys = np.arange(margin, self.size - margin, step)
        for x in xs:
            for y in ys:
                candidates.append((x + self.rng.uniform(-6, 6), y + self.rng.uniform(-6, 6)))
        self.rng.shuffle(candidates)
        for (x, y) in candidates:
            if len(fields) >= n:
                break
            if not self._field_ok_point((x, y), margin):
                continue
            slope = self.sample_slope((x, y))
            width = float(np.clip(21.0 - 0.40 * slope, 9.0, 20.0) * self.rng.uniform(0.85, 1.15))
            length = width * self.rng.uniform(1.9, 3.1)
            ang = self.grad_angle((x, y)) + self.rng.uniform(-0.12, 0.12)
            poly = rect_polygon((x, y), length, width, ang)
            if not self._field_poly_ok(poly, margin):
                continue
            if any(self._polys_overlap(poly, q["polygon"]) for q in fields):
                continue
            fields.append({"polygon": poly, "centroid": np.array([x, y])})
        if len(fields) < max(4, n - 2):
            raise RuntimeError(f"{self.spec.sid}: 只生成 {len(fields)}/{n} 块地，检查场景参数")

        # 按生成顺序编号（基线的固定顺序即调查顺序，不代表空间最优）
        after = []
        for i, f in enumerate(fields, start=1):
            poly = f["polygon"]
            fid = f"T{i:03d}"
            c = polygon_centroid(poly)
            after.append({
                "field_id": fid,
                "polygon": poly,
                "centroid": c,
                "area_m2": float(polygon_area(poly)),
                "mean_elev": float(self.elev_at(c)),
                "mean_slope": float(self.sample_slope(c)),
            })
        self.fields = after

    def _inside(self, p, margin: float) -> bool:
        return margin <= p[0] <= self.size - margin and margin <= p[1] <= self.size - margin

    def _rc(self, p) -> Tuple[int, int]:
        c = int(np.clip(p[0] / self.cell, 0, self.nx - 1))
        r = int(np.clip((self.size - p[1]) / self.cell, 0, self.ny - 1))
        return r, c

    def _field_ok_point(self, p, margin: float) -> bool:
        if not self._inside(p, margin):
            return False
        r, c = self._rc(p)
        if self.slope[r, c] > 30:
            return False
        if self.building_mask[r, c] or self.water_mask[r, c] or self.power_mask[r, c]:
            return False
        if self.road_mask[r, c] or self.forest_mask[r, c]:
            return False
        if self.dist_road[r, c] < 7.0:
            return False
        return True

    def _field_poly_ok(self, poly: np.ndarray, margin: float) -> bool:
        pts = list(poly) + [polygon_centroid(poly)]
        for k in range(4):
            pts.append((poly[k] + poly[(k + 1) % 4]) / 2)
        for p in pts:
            if not self._field_ok_point(p, margin):
                return False
        return True

    def elev_at(self, xy) -> float:
        x, y = float(xy[0]), float(xy[1])
        c = int(np.clip(x / self.cell - 0.5, 0, self.nx - 1))
        r = int(np.clip((self.size - y) / self.cell - 0.5, 0, self.ny - 1))
        return float(self.elevation[r, c])

    # -------------------------------------------------- 树木 / 电线 / 沟渠
    def place_trees(self):
        trees: List[Tuple[np.ndarray, float]] = []
        n = self.spec.n_trees

        # 先在部分地块的长边外侧种 1 棵树，制造真实的“田边树”避让场景
        for idx, f in enumerate(self.fields):
            if idx % 2 == 1:
                continue
            poly = f["polygon"]
            edge = 0 if idx % 4 == 0 else 2
            a, b = poly[edge], poly[(edge + 1) % 4]
            t = self.rng.uniform(0.3, 0.7)
            p = a + (b - a) * t
            center = polygon_centroid(poly)
            outward = p - center
            norm = float(np.linalg.norm(outward))
            if norm < 1e-6:
                continue
            rad = float(self.rng.uniform(2.5, 3.6))
            cand = p + outward / norm * (rad + 2.2)
            r, cc = self._rc(cand)
            if not self._inside(cand, 12) or self.road_mask[r, cc] or self.building_mask[r, cc]:
                continue
            if any(point_in_polygon(cand, g["polygon"]) for g in self.fields):
                continue
            if distance_point_to_polyline(cand, np.vstack([poly, poly[0]])) < rad + 1.5:
                continue
            trees.append((cand, rad))

        attempts = 0
        while len(trees) < n and attempts < 6000:
            attempts += 1
            x = self.rng.uniform(20, self.size - 20); y = self.rng.uniform(20, self.size - 20)
            c = np.array([x, y])
            rad = float(self.rng.uniform(2.5, 4.5))
            r, cc = self._rc(c)
            if self.building_mask[r, cc] or self.road_mask[r, cc]:
                continue
            if self.dist_road[r, cc] < 4:
                continue
            blocked_by_field = False
            for f in self.fields:
                ring = np.vstack([f["polygon"], f["polygon"][0]])
                if point_in_polygon(c, f["polygon"]) or distance_point_to_polyline(c, ring) < rad + 1.5:
                    blocked_by_field = True
                    break
            if blocked_by_field:
                continue
            if any(np.linalg.norm(c - t[0]) < 7 for t in trees):
                continue
            trees.append((c, rad))
        self.trees = trees
        m = np.zeros((self.ny, self.nx), dtype=bool)
        for c, rad in trees:
            rr, cc2 = self._rc(c)
            rad_cells = int(math.ceil(rad / self.cell))
            r0, r1 = max(0, rr - rad_cells), min(self.ny, rr + rad_cells + 1)
            c0, c1 = max(0, cc2 - rad_cells), min(self.nx, cc2 + rad_cells + 1)
            yy, xx = np.mgrid[r0:r1, c0:c1]
            xs = (xx + 0.5) * self.cell; ys = self.size - (yy + 0.5) * self.cell
            m[r0:r1, c0:c1] |= ((xs - c[0]) ** 2 + (ys - c[1]) ** 2) <= rad * rad
        self.tree_mask = m

    def place_powerlines(self):
        lines = []
        for i in range(self.spec.n_powerlines):
            y = self.size * (0.18 + 0.55 * i / max(1, self.spec.n_powerlines))
            x0 = self.size * 0.03
            length = self.size * 0.30
            pts = [(x0, y), (x0 + length * 0.45, y + 18), (x0 + length, y - 10)]
            lines.append(np.array(pts, dtype=float))
        self.powerlines = lines
        self.power_mask = _line_mask(lines, self.nx, self.ny, self.size, self.cell, 4.0)

    def place_ditches(self):
        # 真实水系优先；没有则沿场景南侧布一条仿真沟渠（软风险，不阻挡飞行）
        lines = list(self.osm["waterways"])
        if not lines:
            y = self.size * 0.06
            lines = [np.array([(self.size * 0.05, y), (self.size * 0.55, y + 15), (self.size * 0.95, y)], dtype=float)]
        self.ditches = lines
        self.ditch_mask = _line_mask(lines, self.nx, self.ny, self.size, self.cell, 2.0)

    # -------------------------------------------------- 风险
    def assign_risk(self):
        elevs = np.array([f["mean_elev"] for f in self.fields])
        lo, hi = float(elevs.min()), float(elevs.max())
        vd = []
        for f in self.fields:
            if self.building_centers:
                vd.append(min(float(np.linalg.norm(f["centroid"] - c)) for c in self.building_centers))
            else:
                vd.append(400.0)
        vd = np.array(vd)
        raw = (
            0.55 * (1.0 - (elevs - lo) / max(hi - lo, 1e-6))
            + 0.25 * np.exp(-vd / 180.0)
            + 0.20 * self.rng.random(len(elevs))
        )
        ranks = np.argsort(np.argsort(raw)).astype(float)
        if len(elevs) > 1:
            ranks /= (len(elevs) - 1)
        risk = 0.12 + 0.83 * ranks
        for f, r in zip(self.fields, risk):
            f["risk"] = float(r)
            if r < 0.40:
                f["risk_level"] = "low"; f["priority"] = 1
            elif r < 0.68:
                f["risk_level"] = "medium"; f["priority"] = 2
            else:
                f["risk_level"] = "high"; f["priority"] = 3

    # -------------------------------------------------- 起降点
    def choose_depot(self):
        best = None; best_score = 1e9
        pts = []
        for d in np.arange(25, 70, 5):
            for t in np.arange(25, self.size - 25, 20):
                pts += [(d, t), (self.size - d, t), (t, d), (t, self.size - d)]
        for p in pts:
            r, c = self._rc(p)
            if self.building_mask[r, c] or self.tree_mask[r, c] or self.power_mask[r, c]:
                continue
            if any(point_in_polygon(p, f["polygon"]) for f in self.fields):
                continue
            s = float(self.slope[r, c])
            if s > 22:
                continue
            score = s + 0.02 * float(self.dist_road[r, c]) + 0.001 * float(np.linalg.norm(np.array(p) - self.size / 2))
            if score < best_score:
                best, best_score = np.array(p, dtype=float), score
        if best is None:
            best = np.array([40.0, 40.0], dtype=float)
        self.depot = best

    # -------------------------------------------------- 汇总
    def finalize(self):
        cell = self.cell
        hard = self.building_mask | self.tree_mask | self.power_mask
        soft = np.zeros((self.ny, self.nx), dtype=np.float32)
        soft[self.road_mask] = np.maximum(soft[self.road_mask], 0.9)
        soft[self.track_mask] = np.maximum(soft[self.track_mask], 0.45)
        soft[self.ditch_mask] = np.maximum(soft[self.ditch_mask], 0.7)
        soft[self.forest_mask] = np.maximum(soft[self.forest_mask], 0.4)
        # 地块区域不应被软风险覆盖（地块内部视为农田，不是道路）
        field_id = np.zeros((self.ny, self.nx), dtype=np.int32)
        shapes = []
        for i, f in enumerate(self.fields, start=1):
            shapes.append((_geom_poly(f["polygon"]), i))
            m = _raster_poly(f["polygon"], self.nx, self.ny, self.transform_local)
            soft[m] = 0.0
        if shapes:
            field_id = rasterize(shapes, out_shape=(self.ny, self.nx), transform=self.transform_local,
                                 fill=0, dtype="int32", all_touched=False)
        dist_hard = distance_transform_edt(hard, cell)

        self.hard = hard; self.soft = soft; self.field_id = field_id; self.dist_hard = dist_hard
        self.field_by_id = {f["field_id"]: f for f in self.fields}

    # -------------------------------------------------- 写出
    def write(self):
        out = self.out_dir
        meta = self.meta()
        np.savez_compressed(
            out / "grid.npz",
            elevation=self.elevation, slope=self.slope, aspect=self.aspect,
            roughness=self.roughness, hard=self.hard.astype(np.uint8),
            soft=self.soft, field_id=self.field_id, dist_hard=self.dist_hard,
        )
        self._write_tif(out / "dem.tif", self.elevation, "float32", -9999.0)
        self._write_tif(out / "slope.tif", self.slope, "float32", -9999.0)
        self._write_tif(out / "field_id.tif", self.field_id, "int32", 0)
        self._write_tif(out / "obstacle_mask.tif", self.hard.astype(np.uint8), "uint8", 0)
        self._write_geojson(out, meta)
        self._write_shapefiles(out)
        with open(out / "scene.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        self._render_preview(out / "preview.png")

    def _write_tif(self, path: Path, arr: np.ndarray, dtype: str, nodata):
        with rasterio.open(
            path, "w", driver="GTiff", height=self.ny, width=self.nx, count=1,
            dtype=dtype, crs=CRS_UTM, transform=self.transform_abs,
            nodata=nodata, compress="lzw", tiled=True,
        ) as ds:
            ds.write(arr.astype(dtype), 1)

    def _write_geojson(self, out: Path, meta: Dict):
        def feat(geom, props):
            return {"type": "Feature", "geometry": geom, "properties": props}

        boundary = [[0, 0], [self.size, 0], [self.size, self.size], [0, self.size], [0, 0]]
        fc_b = {"type": "FeatureCollection", "features": [
            feat({"type": "Polygon", "coordinates": [boundary]}, {"name": self.spec.sid, "crs": CRS_UTM})
        ]}
        (out / "boundary.geojson").write_text(json.dumps(fc_b, ensure_ascii=False), encoding="utf-8")

        ffeats = []
        for f in self.fields:
            ring = [[round(float(x), 2), round(float(y), 2)] for x, y in f["polygon"]]
            ring.append(ring[0])
            ffeats.append(feat(
                {"type": "Polygon", "coordinates": [ring]},
                {"field_id": f["field_id"], "risk": round(f["risk"], 3), "risk_level": f["risk_level"],
                 "priority": f["priority"], "area_m2": round(f["area_m2"], 1),
                 "mean_elev": round(f["mean_elev"], 1)},
            ))
        (out / "fields.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": ffeats}, ensure_ascii=False), encoding="utf-8")

        ofeats = []

        def add_line(line, kind: str, source: str, prefix: str):
            for k, seg in enumerate(_clip_line(line, 0.0, self.size)):
                ofeats.append(feat(
                    {"type": "LineString", "coordinates": [[round(float(x), 2), round(float(y), 2)] for x, y in seg]},
                    {"kind": kind, "source": source, "id": f"{prefix}_{k}"}))

        for i, line in enumerate(self.osm["roads"]):
            add_line(line, "road", "OSM", f"road_{i}")
        for i, line in enumerate(self.osm["paths"]):
            add_line(line, "path", "OSM", f"path_{i}")
        for i, line in enumerate(self.ditches):
            add_line(line, "ditch", "OSM" if i < len(self.osm["waterways"]) else "simulated", f"ditch_{i}")
        for i, line in enumerate(self.powerlines):
            add_line(line, "powerline", "simulated", f"power_{i}")
        for i, poly in enumerate(self.buildings):
            ring = [[round(float(x), 2), round(float(y), 2)] for x, y in poly]; ring.append(ring[0])
            ofeats.append(feat({"type": "Polygon", "coordinates": [ring]},
                               {"kind": "building", "source": "simulated", "id": f"bldg_{i}"}))
        for i, (c, rad) in enumerate(self.trees):
            ofeats.append(feat({"type": "Point", "coordinates": [round(float(c[0]), 2), round(float(c[1]), 2)]},
                               {"kind": "tree", "source": "simulated", "radius_m": round(rad, 1), "id": f"tree_{i}"}))
        for i, poly in enumerate(self.osm["forest"]):
            ring = [[round(float(x), 2), round(float(y), 2)] for x, y in poly]; ring.append(ring[0])
            ofeats.append(feat({"type": "Polygon", "coordinates": [ring]},
                               {"kind": "forest", "source": "OSM", "id": f"forest_{i}"}))
        (out / "obstacles.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": ofeats}, ensure_ascii=False), encoding="utf-8")

    def _shp_writer(self, path: Path, shape_type):
        import shapefile
        w = shapefile.Writer(str(path), shapeType=shape_type, encoding="utf-8")
        from rasterio.crs import CRS
        (path.with_suffix(".prj")).write_text(CRS.from_string(CRS_UTM).to_wkt(), encoding="utf-8")
        return w

    def _write_shapefiles(self, out: Path):
        try:
            import shapefile
        except Exception as e:
            (out / "SHAPEFILE_SKIPPED.txt").write_text(f"pyshp 未安装，跳过 shp 导出：{e}", encoding="utf-8")
            return
        ox, oy = float(self.origin[0]), float(self.origin[1])

        def A(p):
            return [round(float(p[0]) + ox, 2), round(float(p[1]) + oy, 2)]

        # 地块（Shapefile 用 UTM 绝对坐标，与 dem.tif 的 .prj 一致）
        w = self._shp_writer(out / "fields.shp", shapefile.POLYGON)
        w.field("field_id", "C", 8); w.field("risk", "N", 8, 3); w.field("level", "C", 8)
        w.field("priority", "N", 3); w.field("area_m2", "N", 12, 1); w.field("elev", "N", 10, 1)
        for f in self.fields:
            w.poly([[A(p) for p in f["polygon"]]])
            w.record(f["field_id"], round(f["risk"], 3), f["risk_level"], f["priority"], round(f["area_m2"], 1), round(f["mean_elev"], 1))
        w.close()
        # 边界
        w = self._shp_writer(out / "boundary.shp", shapefile.POLYGON)
        w.field("name", "C", 40)
        w.poly([[A((0, 0)), A((self.size, 0)), A((self.size, self.size)), A((0, self.size)), A((0, 0))]])
        w.record(self.spec.sid); w.close()
        # 建筑
        w = self._shp_writer(out / "buildings.shp", shapefile.POLYGON)
        w.field("id", "C", 12); w.field("source", "C", 12)
        for i, poly in enumerate(self.buildings):
            w.poly([[A(p) for p in poly]]); w.record(f"bldg_{i}", "simulated")
        w.close()
        # 树木
        w = self._shp_writer(out / "trees.shp", shapefile.POINT)
        w.field("id", "C", 12); w.field("radius_m", "N", 6, 1)
        for i, (c, rad) in enumerate(self.trees):
            w.point(float(c[0]) + ox, float(c[1]) + oy); w.record(f"tree_{i}", round(rad, 1))
        w.close()
        # 道路 / 沟渠 / 电线
        for name, lines, src in [("roads", self.osm["roads"], "OSM"), ("ditches", self.ditches, "mixed"), ("powerlines", self.powerlines, "simulated")]:
            w = self._shp_writer(out / f"{name}.shp", shapefile.POLYLINE)
            w.field("id", "C", 12); w.field("source", "C", 12)
            for i, line in enumerate(lines):
                for k, seg in enumerate(_clip_line(line, 0.0, self.size)):
                    w.line([[A(p) for p in seg]])
                    w.record(f"{name}_{i}_{k}", src)
            w.close()

    def _render_preview(self, path: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(self.elevation, vert_exag=2.0, dx=self.cell, dy=self.cell)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(hs, cmap="gray", origin="upper", extent=[0, self.size, 0, self.size])
        cmap = {"low": "#2e7d32", "medium": "#f9a825", "high": "#c62828"}
        for f in self.fields:
            p = f["polygon"]
            ax.fill(p[:, 0], p[:, 1], color=cmap[f["risk_level"]], alpha=0.55, edgecolor="white", linewidth=0.6)
            ax.text(f["centroid"][0], f["centroid"][1], f["field_id"], fontsize=5, ha="center", va="center", color="white")
        hr, hc = np.where(self.hard)
        if len(hr):
            ax.scatter(hc * self.cell, self.size - hr * self.cell, s=0.15, c="#37474f", alpha=0.6, linewidths=0)
        for line in self.osm["roads"]:
            ax.plot(line[:, 0], line[:, 1], color="#ffd54f", linewidth=0.8)
        for line in self.powerlines:
            ax.plot(line[:, 0], line[:, 1], color="#ff1744", linewidth=1.2, linestyle="--")
        ax.plot(self.depot[0], self.depot[1], marker="*", color="#00e5ff", markersize=14, markeredgecolor="black")
        ax.set_xlim(0, self.size); ax.set_ylim(0, self.size)
        ax.set_title(f"{self.spec.sid}  {self.spec.name}")
        ax.set_xlabel("x / m (east)"); ax.set_ylabel("y / m (north)")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

    # -------------------------------------------------- 元数据
    def meta(self) -> Dict:
        field_area = sum(f["area_m2"] for f in self.fields)
        lon, lat = self.spec.center_lon, self.spec.center_lat
        uav = DEFAULT_UAV
        return {
            "scene_id": self.spec.sid,
            "name": self.spec.name,
            "complexity": self.spec.complexity,
            "cell_m": self.cell,
            "nx": self.nx, "ny": self.ny, "size_m": self.size,
            "origin_xy": [round(self.origin[0], 2), round(self.origin[1], 2)],
            "bbox_utm": [round(self.origin[0], 2), round(self.origin[1], 2),
                         round(self.origin[0] + self.size, 2), round(self.origin[1] + self.size, 2)],
            "crs": CRS_UTM,
            "center_lonlat": [lon, lat],
            "depot_xy": [round(float(self.depot[0]), 2), round(float(self.depot[1]), 2)],
            "uav": {
                "cruise_speed": uav.cruise_speed, "climb_speed": uav.climb_speed,
                "alt_agl": uav.alt_agl, "fov_deg": uav.fov_deg,
                "battery_wh": uav.battery_wh, "reserve_ratio": uav.reserve_ratio,
                "footprint_radius_m": round(uav.footprint_radius, 2),
                "gsd_cm": round(uav.gsd_cm, 2),
            },
            "fields": [
                {"field_id": f["field_id"], "risk": round(f["risk"], 3), "risk_level": f["risk_level"],
                 "priority": f["priority"], "area_m2": round(f["area_m2"], 1),
                 "mean_elev": round(f["mean_elev"], 1),
                 "centroid": [round(float(f["centroid"][0]), 2), round(float(f["centroid"][1]), 2)],
                 "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in f["polygon"]]}
                for f in self.fields
            ],
            "boundary": [[0, 0], [self.size, 0], [self.size, self.size], [0, self.size]],
            "counts": {
                "n_fields": len(self.fields),
                "relief_m": round(float(self.elevation.max() - self.elevation.min()), 1),
                "mean_slope_deg": round(float(self.slope.mean()), 1),
                "field_area_m2": round(field_area, 1),
                "field_area_pct": round(100 * field_area / (self.size ** 2), 2),
                "n_buildings": len(self.buildings) + len(self.osm_building_polys),
                "n_trees": len(self.trees),
                "n_powerlines": len(self.powerlines),
                "n_roads": len(self.osm["roads"]),
                "n_waterways": len(self.osm["waterways"]),
                "hard_area_pct": round(100 * float(self.hard.mean()), 2),
            },
            "sources": {
                "dem": {
                    "dataset": "Copernicus DEM GLO-30 (COP-DEM_GLO-30-DGED)",
                    "tile": "Copernicus_DSM_COG_10_N23_00_E102_00_DEM",
                    "provider": "ESA / Copernicus Programme，AWS Open Data 镜像",
                    "url": "https://copernicus-dem-30m.s3.amazonaws.com/",
                    "license": "Copernicus DEM 免费开放（详见数据集存储/README.md）",
                    "note": "30 m 原始 DEM 双线性重采样到 5 m 规划网格，未做地形改造",
                },
                "osm": {
                    "dataset": "OpenStreetMap（道路、水系、林地、农田等）",
                    "provider": "OpenStreetMap contributors / Overpass API",
                    "license": "ODbL 1.0",
                    "extract_bbox_lonlat": [102.690, 23.030, 102.782, 23.132],
                    "note": "仅使用与场景相交的要素；footway/path 不作为飞行障碍",
                },
                "fields": "人工生成：沿局部等高线方向布设的矩形梯田地块（仿真）",
                "obstacles": "OSM 道路/水系/林地（真实）+ 建筑、树木、电线（人工布设，仿真）",
                "risk": "仿真病害风险场：低海拔、靠近村庄的地块风险更高，固定随机种子生成",
            },
            "notes": [
                "DEM 为真实公开数据；地块边界、建筑、树木、电线、风险值均为仿真，不可当作实测结果。",
                "30 m DEM 无法表达田埂与沟渠细节，因此地块与障碍物按 5 m 网格人工布设，符合“真实地形底图 + 局部仿真”的约定。",
                "场景坐标：x 向东、y 向北，原点为场景西南角，单位 m；GeoTIFF 使用 EPSG:32648。",
                "所有随机过程使用固定种子，可完全复现。",
            ],
            "seed": self.spec.seed,
        }

    # -------------------------------------------------- 主流程
    def build(self):
        self.prepare_terrain()
        self.prepare_osm()
        self.place_buildings()
        self.place_powerlines()
        self.place_fields()
        self.place_trees()
        self.place_ditches()
        self.assign_risk()
        self.choose_depot()
        self.finalize()
        self.write()
        return self.meta()


def build_all(dem_path: Path, osm_path: Optional[Path], out_root: Path, scene_ids: Optional[Sequence[str]] = None):
    results = []
    for sid in (scene_ids or list(SCENE_SPECS.keys())):
        spec = SCENE_SPECS[sid]
        builder = SceneBuilder(spec, dem_path, osm_path, Path(out_root) / sid)
        meta = builder.build()
        results.append(meta)
        print(f"[OK] {sid}: {meta['counts']['n_fields']} fields, "
              f"relief {meta['counts']['relief_m']} m, hard {meta['counts']['hard_area_pct']}%")
    return results
