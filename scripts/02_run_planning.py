# -*- coding: utf-8 -*-
"""交付物2：运行基线与改进路径规划，输出固定格式航迹 JSON。

用法：
    python scripts/02_run_planning.py                  # 3 场景 × 2 方法
    python scripts/02_run_planning.py --scenes scene_medium --methods improved
输出：result/traj/{scene}_{method}.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import SCENE_IDS, TRAJ_DIR
from src.planner.route import plan_mission, save_traj
from src.simulator.scene import Scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=["baseline", "improved"])
    args = ap.parse_args()
    scenes = args.scenes or SCENE_IDS
    for sid in scenes:
        scene = Scene.load(sid)
        for method in args.methods:
            t0 = time.perf_counter()
            res = plan_mission(scene, method=method)
            path = save_traj(res, TRAJ_DIR)
            m = res["metrics"]
            print(f"[{sid}/{method}] {len(res['fields'])} 块地, 航程 {m['path_length_m']:.0f} m, "
                  f"时间 {m['flight_time_s']:.0f} s, 能耗 {m['energy_wh']:.1f} Wh, "
                  f"规划 {time.perf_counter()-t0:.2f} s -> {path}")


if __name__ == "__main__":
    main()
