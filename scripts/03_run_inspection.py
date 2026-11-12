# -*- coding: utf-8 -*-
"""交付物3：巡检仿真，计算覆盖率、地块访问顺序、图像索引。

用法：
    python scripts/03_run_inspection.py
    python scripts/03_run_inspection.py --no-images      # 只算指标，不生成占位图
输出：
    result/inspect/{scene}_{method}_inspection.json
    result/inspect/{scene}_{method}_fields.csv
    result/inspect/images/...   （仿真占位图，内含 SIMULATED 水印）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import INSPECT_DIR, TRAJ_DIR
from src.simulator.inspect import inspect_mission, save_inspection
from src.simulator.scene import Scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default=str(TRAJ_DIR))
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    files = sorted(Path(args.traj_dir).glob("*_*.json"))
    if not files:
        raise SystemExit("没有航迹文件，请先运行 scripts/02_run_planning.py")
    for fp in files:
        traj = json.loads(fp.read_text(encoding="utf-8"))
        traj["_path"] = str(fp)
        scene = Scene.load(traj["scene"])
        res = inspect_mission(traj, scene, generate_images=not args.no_images)
        save_inspection(res, traj, INSPECT_DIR)
        t = res["totals"]
        print(f"[{traj['scene']}/{traj['method']}] 覆盖率 {t['coverage_pct']:.1f}% "
              f"(地块均值 {t['mean_field_coverage_pct']:.1f}%), 图像 {t['n_images']} 张, "
              f"访问顺序 {'->'.join(res['visit_order'][:5])}...")


if __name__ == "__main__":
    main()
