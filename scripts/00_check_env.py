# -*- coding: utf-8 -*-
"""运行前自检：依赖、数据、场景与结果文件是否齐全。

用法：python scripts/00_check_env.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import DATASET_DIR, EXP_DIR, INSPECT_DIR, SCENE_DIR, SCENE_IDS, TRAJ_DIR


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main():
    print(f"Python : {sys.version.split()[0]}  ({sys.executable})")
    print(f"项目根 : {ROOT}")
    print("\n[依赖]")
    required = {"numpy": "数组计算", "matplotlib": "图表", "pandas": "表格（可选）"}
    optional = {"rasterio": "读取/重建 DEM 场景（仅重建场景需要）", "shapefile": "导出 shp（仅重建场景需要）",
                "fastapi": "网页接口", "uvicorn": "启动接口", "PIL": "仿真占位图"}
    ok = True
    for m, why in required.items():
        flag = has(m)
        ok &= flag
        print(f"  {'OK ' if flag else 'MISS'}  {m:12s} {why}")
    for m, why in optional.items():
        flag = has(m)
        print(f"  {'OK ' if flag else '-- '}  {m:12s} {why}")
    print("\n[数据与结果]")
    dem = list((DATASET_DIR / "Copernicus_DEM_GLO30" / "raw").glob("*.tif"))
    print(f"  {'OK ' if dem else 'MISS'}  DEM 原始瓦片：{len(dem)} 个")
    for sid in SCENE_IDS:
        d = SCENE_DIR / sid
        print(f"  {'OK ' if (d/'scene.json').exists() else 'MISS'}  场景 {sid}")
    for sid in SCENE_IDS:
        for m in ("baseline", "improved"):
            f = TRAJ_DIR / f"{sid}_{m}.json"
            print(f"  {'OK ' if f.exists() else '-- '}  航迹 {sid}_{m}")
    print(f"  {'OK ' if (EXP_DIR/'metrics.csv').exists() else '-- '}  实验指标 metrics.csv")
    print(f"  {'OK ' if (INSPECT_DIR).exists() else '-- '}  巡检结果目录")
    print("\n结论：", "环境可用" if ok else "缺少必需依赖，请先 pip install -r requirements.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
