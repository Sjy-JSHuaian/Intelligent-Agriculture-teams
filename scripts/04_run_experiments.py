# -*- coding: utf-8 -*-
"""交付物4：批量算法实验（3 场景 × 基线/改进）+ 局部重规划实验。

用法：
    python scripts/04_run_experiments.py
输出：
    result/experiment/metrics.csv     主实验表（路径长度/覆盖率/规划时间/安全/能耗…）
    result/experiment/metrics.json
    result/experiment/replan.csv      禁飞区重规划实验
    result/experiment/replan.json
同时刷新 result/traj 与 result/inspect。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import EXP_DIR, INSPECT_DIR, SCENE_IDS, TRAJ_DIR
from src.planner.route import plan_mission, save_traj
from src.simulator.inspect import inspect_mission, save_inspection
from src.simulator.replan import run_replan
from src.simulator.scene import Scene

COLS = [
    "scene", "complexity", "method", "n_fields", "path_length_m", "flight_time_s",
    "energy_wh", "battery_pct_used", "battery_reserve_ok", "planning_time_s",
    "coverage_pct", "mean_field_coverage_pct", "n_images", "min_clearance_m",
    "mean_clearance_m", "near_obstacle_length_m", "safety_violation_m", "collision_count",
    "high_risk_mean_order", "weighted_latency_s", "n_waypoints",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--no-replan", action="store_true")
    args = ap.parse_args()
    scenes = args.scenes or SCENE_IDS

    rows = []
    replan_rows = []
    replan_all = {}
    for sid in scenes:
        scene = Scene.load(sid)
        for method in ("baseline", "improved"):
            t0 = time.perf_counter()
            traj = plan_mission(scene, method=method)
            tp = save_traj(traj, TRAJ_DIR)
            traj["_path"] = tp
            insp = inspect_mission(traj, scene, generate_images=True)
            save_inspection(insp, traj, INSPECT_DIR)
            m = traj["metrics"]
            m["coverage_pct"] = insp["totals"]["coverage_pct"]
            rows.append({
                "scene": sid, "complexity": scene.complexity, "method": method,
                "n_fields": len(traj["fields"]), "path_length_m": m["path_length_m"],
                "flight_time_s": m["flight_time_s"], "energy_wh": m["energy_wh"],
                "battery_pct_used": m["battery_pct_used"], "battery_reserve_ok": m["battery_reserve_ok"],
                "planning_time_s": m["planning_time_s"], "coverage_pct": insp["totals"]["coverage_pct"],
                "mean_field_coverage_pct": insp["totals"]["mean_field_coverage_pct"],
                "n_images": insp["totals"]["n_images"], "min_clearance_m": m["min_clearance_m"],
                "mean_clearance_m": m["mean_clearance_m"], "near_obstacle_length_m": m["near_obstacle_length_m"], "safety_violation_m": m["safety_violation_m"],
                "collision_count": m["collision_count"], "high_risk_mean_order": m["high_risk_mean_order"],
                "weighted_latency_s": m["weighted_latency_s"], "n_waypoints": m["n_waypoints"],
            })
            print(f"[{sid}/{method}] {m['path_length_m']:.0f} m, {m['energy_wh']:.1f} Wh, "
                  f"覆盖 {insp['totals']['coverage_pct']:.1f}%, 规划 {m['planning_time_s']:.2f} s")

        if not args.no_replan:
            traj = json.loads((TRAJ_DIR / f"{sid}_improved.json").read_text(encoding="utf-8"))
            r = run_replan(scene, traj)
            if r is None:
                print(f"[{sid}] 重规划：未找到可行禁飞区位置")
                continue
            replan_all[sid] = r
            replan_rows.append({
                "scene": sid, "nfz_radius_m": r["nfz"]["radius_m"],
                "injected_fraction": r["nfz"]["injected_at_fraction"],
                "skipped_fields": len(r["skipped_fields"]),
                "no_replan_intrusion_m": r["no_replan"]["intrusion_m"],
                "replan_intrusion_m": r["replan"]["intrusion_m"],
                "extra_distance_m": r["replan"]["extra_distance_m"],
                "total_distance_m": r["replan"]["total_distance_m"],
                "replan_time_s": r["replan"]["replan_time_s"],
                "remaining_fields": r["n_remaining_fields"],
            })
            print(f"[{sid}/replan] 禁飞区 {r['nfz']['radius_m']:.0f} m, 原航迹闯入 "
                  f"{r['no_replan']['intrusion_m']:.1f} m -> 重规划后 {r['replan']['intrusion_m']:.1f} m, "
                  f"跳过 {len(r['skipped_fields'])} 块")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXP_DIR / "metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    (EXP_DIR / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    if replan_rows:
        with open(EXP_DIR / "replan.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(replan_rows[0].keys())); w.writeheader(); w.writerows(replan_rows)
        (EXP_DIR / "replan.json").write_text(json.dumps(replan_all, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n已写出：{EXP_DIR / 'metrics.csv'}")
    if replan_rows:
        print(f"已写出：{EXP_DIR / 'replan.csv'}")


if __name__ == "__main__":
    main()

