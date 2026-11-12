# -*- coding: utf-8 -*-
"""交付物1：构建 3 套梯田仿真场景（读取真实 DEM + OSM，输出 5 m 栅格/矢量）。

用法（在 project_A 目录下）：
    python scripts/01_build_scenes.py
    python scripts/01_build_scenes.py --scenes scene_medium
输出：
    data/scenes/scene_*/dem.tif, slope.tif, field_id.tif, obstacle_mask.tif,
    grid.npz, scene.json, fields.geojson, obstacles.geojson, *.shp, preview.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_DIR, SCENE_DIR, SCENE_SPECS
from src.simulator.scene_builder import build_all


def find_dem() -> Path:
    raw = DATASET_DIR / "Copernicus_DEM_GLO30" / "raw"
    cands = sorted(raw.glob("Copernicus_DSM_COG_10_*_DEM.tif"))
    if not cands:
        cands = sorted(raw.glob("*.tif"))
    if not cands:
        raise SystemExit(
            f"未找到 DEM。请先运行 数据集存储/scripts/download_all.py，或把 GLO-30 GeoTIFF 放到：{raw}"
        )
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", type=str, default=None, help="原始 GLO-30 DEM 路径")
    ap.add_argument("--osm", type=str, default=None, help="OSM Overpass JSON 路径")
    ap.add_argument("--scenes", type=str, nargs="*", default=None, help="只构建指定场景")
    ap.add_argument("--out", type=str, default=str(SCENE_DIR))
    args = ap.parse_args()

    dem = Path(args.dem) if args.dem else find_dem()
    osm = Path(args.osm) if args.osm else DATASET_DIR / "OpenStreetMap" / "raw" / "osm_yuanyang_core.json"
    if not dem.exists():
        raise SystemExit(f"DEM 不存在：{dem}")
    if not osm.exists():
        print(f"[警告] OSM 文件不存在（{osm}），将只使用 DEM + 全人工障碍物。")
        osm = None
    print(f"DEM : {dem}")
    print(f"OSM : {osm}")
    metas = build_all(dem, osm, Path(args.out), args.scenes)
    print("\n场景构建完成：")
    for m in metas:
        c = m["counts"]
        print(f"  {m['scene_id']:13s} 地块 {c['n_fields']:2d} | 高差 {c['relief_m']:6.1f} m | "
              f"平均坡度 {c['mean_slope_deg']:4.1f}° | 障碍占比 {c['hard_area_pct']:4.1f}% | "
              f"地块面积 {c['field_area_pct']:4.1f}%")


if __name__ == "__main__":
    main()
