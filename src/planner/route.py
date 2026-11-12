# -*- coding: utf-8 -*-
"""全局巡检路径规划：地块排序 + 覆盖航带 + A* 转场 + 时间/能耗仿真。

基线：地块按编号固定顺序，每块固定航带变体，纯距离 A*。
改进：高风险地块优先排序，按进入/离开位置选择航带变体，
      转场使用带坡度、安全距离、软风险与爬升代价的 A*。
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.config import (
    DEFAULT_PLAN, DEFAULT_UAV, DEFAULT_WEIGHTS, SCHEMA_VERSION, CRS_UTM,
    CostWeights, PlanParams, UAVSpec,
)
from src.common.geometry import polyline_length
from src.planner.astar import astar, path_clearance_stats
from src.planner.coverage import CoveragePlan, choose_variant, variants
from src.simulator.scene import Field, Scene

TZ = timezone(timedelta(hours=8))


def order_baseline(fields: Sequence[Field], depot: np.ndarray) -> List[Field]:
    return sorted(fields, key=lambda f: f.field_id)


def order_improved(fields: Sequence[Field], depot: np.ndarray) -> List[Field]:
    """先按风险等级从高到低，同等级内最近邻。"""
    order: List[Field] = []
    cur = np.asarray(depot, dtype=float)
    for tier in (2, 1, 0):
        group = [f for f in fields if f.risk_tier == tier]
        while group:
            nxt = min(group, key=lambda f: float(np.linalg.norm(f.centroid - cur)))
            order.append(nxt)
            group.remove(nxt)
            cur = nxt.centroid
    return order


def _energy_leg(uav: UAVSpec, dh: float, dz: float) -> float:
    e = (uav.p_cruise_w + uav.p_avionics_w) * dh / uav.cruise_speed / 3600.0
    if dz > 0:
        e += (uav.p_climb_w + uav.p_avionics_w) * dz / uav.climb_speed / 3600.0
    elif dz < 0:
        e += (uav.p_cruise_w * 0.6 + uav.p_avionics_w) * (-dz) / uav.climb_speed / 3600.0
    return e


def _heading(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(b) - np.asarray(a)
    return float(math.degrees(math.atan2(d[0], d[1])) % 360.0)  # 0=北, 90=东


def plan_mission(
    scene: Scene,
    method: str = "improved",
    uav: Optional[UAVSpec] = None,
    params: PlanParams = DEFAULT_PLAN,
    weights: CostWeights = DEFAULT_WEIGHTS,
    start_xy: Optional[Sequence[float]] = None,
    field_subset: Optional[Sequence[Field]] = None,
    blocked_extra: Optional[np.ndarray] = None,
    start_action: str = "takeoff",
) -> Dict:
    if method not in ("baseline", "improved"):
        raise ValueError(f"unknown method: {method}")
    uav = uav or scene.uav()
    t_start = time.perf_counter()

    pool = list(field_subset) if field_subset is not None else list(scene.fields)
    fields = order_improved(pool, scene.depot) if method == "improved" else order_baseline(pool, scene.depot)
    depot = np.asarray(scene.depot, dtype=float)
    start = np.asarray(start_xy, dtype=float) if start_xy is not None else depot
    alt = uav.alt_agl

    def z_of(xy):
        return scene.elevation_at(xy) + alt

    waypoints: List[Dict] = []
    seq = 0
    t = 0.0
    energy = 0.0
    distance = 0.0

    def add(xy, action, field_id=None, dwell=0.0, heading=None):
        nonlocal seq, t, energy, distance
        xy = np.asarray(xy, dtype=float)
        z = z_of(xy)
        if waypoints:
            prev = waypoints[-1]
            dh = float(np.linalg.norm(xy - np.array([prev["x"], prev["y"]])))
            dz = z - prev["z"]
            distance += dh
            t += dh / uav.cruise_speed + abs(dz) / uav.climb_speed
            energy += _energy_leg(uav, dh, dz)
        if dwell > 0:
            t += dwell
            energy += (uav.p_hover_w + uav.p_avionics_w) * dwell / 3600.0
        if heading is None:
            heading = 0.0
        waypoints.append({
            "seq": seq, "x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2),
            "z": round(float(z), 2), "t": round(float(t), 2),
            "heading": round(float(heading), 1),
            "field_id": field_id, "action": action,
        })
        seq += 1

    # 起飞 / 重规划起点
    if start_action == "takeoff":
        t += uav.takeoff_s
        energy += (uav.p_hover_w + uav.p_avionics_w) * uav.takeoff_s / 3600.0
        add(start, "takeoff")
    else:
        add(start, "replan_start")
    prev_xy = start

    transit_paths: List[List[Tuple[float, float]]] = []
    field_records: List[Dict] = []

    for i, f in enumerate(fields):
        next_xy = fields[i + 1].centroid if i + 1 < len(fields) else depot
        plans = variants(f, uav, params.overlap)
        if method == "improved":
            cp = choose_variant(plans, prev_xy, next_xy)
        else:
            cp = plans[0]

        leg = astar(scene, prev_xy, cp.entry, mode=method, weights=weights, blocked_extra=blocked_extra)
        path = leg["path"]
        transit_paths.append(path)
        for j, p in enumerate(path):
            if j == 0 and waypoints and abs(p[0] - waypoints[-1]["x"]) < 1e-6 and abs(p[1] - waypoints[-1]["y"]) < 1e-6:
                continue
            hd = _heading(prev_xy, np.asarray(p))
            add(p, "transit", heading=hd)
            prev_xy = np.asarray(p)

        # 进入地块：航带拍照点
        np_prev = np.asarray(prev_xy, dtype=float)
        for k, wp in enumerate(cp.waypoints):
            if k == 0 and float(np.linalg.norm(wp - np_prev)) < 1e-6:
                pass
            hd = _heading(np_prev, wp)
            add(wp, "capture", field_id=f.field_id, dwell=uav.dwell_s, heading=hd)
            prev_xy = wp
            np_prev = wp

        field_records.append({
            "field_id": f.field_id,
            "risk": round(float(f.risk), 3),
            "risk_level": f.risk_level,
            "priority": int(f.priority),
            "visit_order": i + 1,
            "entry": [round(float(cp.entry[0]), 2), round(float(cp.entry[1]), 2)],
            "exit": [round(float(cp.exit[0]), 2), round(float(cp.exit[1]), 2)],
            "n_strips": cp.n_strips,
            "variant": cp.variant,
            "area_m2": round(float(f.area_m2), 1),
        })

    # 返航
    leg = astar(scene, prev_xy, depot, mode=method, weights=weights, blocked_extra=blocked_extra)
    transit_paths.append(leg["path"])
    for j, p in enumerate(leg["path"]):
        if j == 0 and abs(p[0] - waypoints[-1]["x"]) < 1e-6 and abs(p[1] - waypoints[-1]["y"]) < 1e-6:
            continue
        add(p, "transit", heading=_heading(prev_xy, np.asarray(p)))
        prev_xy = np.asarray(p)
    t += uav.landing_s
    energy += (uav.p_hover_w + uav.p_avionics_w) * uav.landing_s / 3600.0
    add(depot, "land")
    waypoints[-1]["z"] = round(float(scene.elevation_at(depot)), 2)
    waypoints[-1]["t"] = round(float(t), 2)

    planning_time = time.perf_counter() - t_start
    parts = [path_clearance_stats(scene, p) for p in transit_paths if len(p) > 1]
    if parts:
        n_tot = sum(x.get("n_samples", 0) for x in parts)
        stats = {
            "min_clearance_m": min(x["min_clearance_m"] for x in parts),
            "mean_clearance_m": sum(x["mean_clearance_m"] * x.get("n_samples", 0) for x in parts) / max(n_tot, 1),
            "violation_length_m": sum(x["violation_length_m"] for x in parts),
            "near_obstacle_length_m": sum(x.get("near_obstacle_length_m", 0.0) for x in parts),
            "obstacle_hits": sum(x["obstacle_hits"] for x in parts),
        }
    else:
        stats = {"min_clearance_m": 0.0, "mean_clearance_m": 0.0, "violation_length_m": 0.0, "near_obstacle_length_m": 0.0, "obstacle_hits": 0}
    transit_stats = stats

    arrival = {r["field_id"]: r for r in field_records}
    high = [r for r in field_records if r["risk_level"] == "high"]
    weights_sum = sum(1.0 + params.risk_weight * r["risk"] for r in field_records)
    weighted_latency = sum(
        (1.0 + params.risk_weight * r["risk"]) * 0.0 for r in field_records
    )
    # 用到达各地块首张照片的时间计算加权延迟
    first_time = {}
    for w in waypoints:
        if w["action"] == "capture" and w["field_id"] not in first_time:
            first_time[w["field_id"]] = w["t"]
    weighted_latency = sum((1.0 + params.risk_weight * r["risk"]) * first_time.get(r["field_id"], t) for r in field_records) / max(weights_sum, 1e-9)

    metrics = {
        "method": method,
        "path_length_m": round(float(distance), 1),
        "flight_time_s": round(float(t), 1),
        "energy_wh": round(float(energy), 2),
        "battery_pct_used": round(float(energy / uav.battery_wh * 100.0), 1),
        "battery_reserve_ok": bool(energy <= uav.battery_wh * (1.0 - uav.reserve_ratio)),
        "planning_time_s": round(float(planning_time), 3),
        "n_waypoints": len(waypoints),
        "n_fields": len(field_records),
        "n_images": sum(1 for w in waypoints if w["action"] == "capture"),
        "high_risk_mean_order": round(float(np.mean([r["visit_order"] for r in high])), 2) if high else None,
        "weighted_latency_s": round(float(weighted_latency), 1),
        "min_clearance_m": round(float(stats["min_clearance_m"]), 2),
        "mean_clearance_m": round(float(stats["mean_clearance_m"]), 2),
        "collision_count": int(stats["obstacle_hits"]),
        "safety_violation_m": round(float(transit_stats["violation_length_m"]), 1),
        "near_obstacle_length_m": round(float(transit_stats.get("near_obstacle_length_m", 0.0)), 1),
        "coverage_pct": None,
        "coverage_pct_mean_field": None,
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene.scene_id,
        "scene_name": scene.name,
        "method": method,
        "created_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "coords": {"crs": CRS_UTM, "local_origin_xy": [0.0, 0.0], "utm_origin_xy": scene.utm_origin.tolist(),
                   "unit": "m", "note": "航迹为局部平面坐标：x 向东，y 向北，原点为场景西南角"},
        "uav": {
            "cruise_speed_mps": uav.cruise_speed, "climb_speed_mps": uav.climb_speed,
            "alt_agl_m": uav.alt_agl, "fov_deg": uav.fov_deg,
            "footprint_radius_m": round(uav.footprint_radius, 2), "gsd_cm": round(uav.gsd_cm, 2),
            "battery_wh": uav.battery_wh,
        },
        "route": [f.field_id for f in fields],
        "fields": field_records,
        "waypoints": waypoints,
        "metrics": metrics,
    }
    return result


def save_traj(result: Dict, out_dir) -> str:
    import json
    from pathlib import Path
    out = Path(out_dir) / f"{result['scene']}_{result['method']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if not k.startswith("_")}, f, ensure_ascii=False, indent=1)
    return str(out)

