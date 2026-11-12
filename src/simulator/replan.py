# -*- coding: utf-8 -*-
"""局部避障与重规划：飞行途中注入临时禁飞区（NFZ），从当前位置重规划剩余地块。"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.config import DEFAULT_PLAN, DEFAULT_WEIGHTS
from src.planner.astar import path_hits_mask
from src.planner.route import _energy_leg, plan_mission
from src.simulator.scene import Scene


def circle_mask(scene: Scene, center: Sequence[float], radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:scene.ny, 0:scene.nx]
    xs = (xx + 0.5) * scene.cell
    ys = scene.size_m - (yy + 0.5) * scene.cell
    return ((xs - center[0]) ** 2 + (ys - center[1]) ** 2) <= radius * radius


def _cumulative(points: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.array([])
    return np.concatenate([[0.0], np.cumsum(d)])


def _pick_nfz(scene: Scene, traj: Dict, fraction: float, radius: float, strict: bool = True):
    from src.common.geometry import distance_point_to_polyline, point_in_polygon

    wps = traj["waypoints"]
    pts = np.array([[w["x"], w["y"]] for w in wps], dtype=float)
    cum = _cumulative(pts)
    target = cum[-1] * fraction
    order = np.argsort(np.abs(cum - target))
    for idx in order:
        idx = int(idx)
        if wps[idx]["action"] != "transit":
            continue
        if idx <= 2 or idx >= len(wps) - 4:
            continue
        p = pts[idx]
        if np.linalg.norm(p - scene.depot) < radius * 2.5:
            continue
        t = pts[min(idx + 3, len(pts) - 1)] - pts[max(idx - 3, 0)]
        norm = float(np.linalg.norm(t))
        if norm < 1e-6:
            continue
        tangent = t / norm
        normal = np.array([-tangent[1], tangent[0]])
        center = p + normal * (0.55 * radius)
        dists = np.linalg.norm(pts - center, axis=1)
        inside = dists < radius
        if not inside.any():
            continue
        first = int(np.argmax(inside))
        cut = first - 1
        if cut <= 1 or cut >= len(wps) - 3:
            continue
        if np.linalg.norm(pts[cut] - center) < radius + 2:
            continue
        if strict:
            bad = False
            for f in scene.fields:
                ring = np.vstack([f.polygon, f.polygon[0]])
                if point_in_polygon(center, f.polygon) or distance_point_to_polyline(center, ring) < radius + 7:
                    bad = True
                    break
            if bad:
                continue
        return center, cut
    return None


def _prefix_energy(uav, wps: Sequence[Dict]) -> float:
    e = 0.0
    for a, b in zip(wps[:-1], wps[1:]):
        dh = float(np.hypot(b["x"] - a["x"], b["y"] - a["y"]))
        dz = float(b["z"] - a["z"])
        e += _energy_leg(uav, dh, dz)
        if b["action"] == "capture":
            e += (uav.p_hover_w + uav.p_avionics_w) * uav.dwell_s / 3600.0
    return e


def run_replan(scene: Scene, traj: Dict, method: str = "improved") -> Optional[Dict]:
    uav = scene.uav()
    wps = traj["waypoints"]
    pts = np.array([[w["x"], w["y"]] for w in wps], dtype=float)
    original_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    from src.common.geometry import distance_point_to_polyline, point_in_polygon

    last_err = None
    for strict in (True, False):
        for fraction, radius in [(0.45, 45.0), (0.40, 40.0), (0.50, 35.0), (0.45, 30.0), (0.35, 28.0)]:
            picked = _pick_nfz(scene, traj, fraction, radius, strict=strict)
            if picked is None:
                continue
            center, cut_idx = picked
            mask = circle_mask(scene, center, radius)
            cut_seq = wps[cut_idx]["seq"]

            skipped = []
            for f in scene.fields:
                ring = np.vstack([f.polygon, f.polygon[0]])
                if point_in_polygon(center, f.polygon) or distance_point_to_polyline(center, ring) < radius:
                    skipped.append(f)
            skip_ids = {f.field_id for f in skipped}
            remaining = [f for f in scene.fields if f.field_id not in skip_ids and any(
                w["field_id"] == f.field_id and w["seq"] > cut_seq for w in wps
            )]
            if len(remaining) < 3 or len(skipped) > 2:
                continue
            try:
                new = plan_mission(
                    scene, method=method, uav=uav, params=DEFAULT_PLAN, weights=DEFAULT_WEIGHTS,
                    start_xy=(wps[cut_idx]["x"], wps[cut_idx]["y"]),
                    field_subset=remaining, blocked_extra=mask, start_action="replan",
                )
            except Exception as e:
                last_err = str(e)
                continue

            prefix = wps[: cut_idx + 1]
            new_wps = new["waypoints"]
            if new_wps and abs(new_wps[0]["x"] - prefix[-1]["x"]) < 1e-6 and abs(new_wps[0]["y"] - prefix[-1]["y"]) < 1e-6:
                new_wps = new_wps[1:]
            t0 = prefix[-1]["t"]
            seq0 = prefix[-1]["seq"] + 1
            stitched = [dict(w) for w in prefix]
            for w in new_wps:
                w2 = dict(w)
                w2["seq"] = w2["seq"] + seq0
                w2["t"] = round(w2["t"] + t0, 2)
                stitched.append(w2)

            stitched_pts = np.array([[w["x"], w["y"]] for w in stitched], dtype=float)
            stitched_len = float(np.linalg.norm(np.diff(stitched_pts, axis=0), axis=1).sum())
            intr_no = path_hits_mask(scene, pts, mask)
            intr_yes = path_hits_mask(scene, stitched_pts, mask)

            return {
                "scene": scene.scene_id,
                "method": method,
                "nfz": {"center": [round(float(center[0]), 1), round(float(center[1]), 1)], "radius_m": radius,
                        "injected_at_fraction": fraction},
                "cut": {"seq": int(cut_seq), "x": wps[cut_idx]["x"], "y": wps[cut_idx]["y"], "t": wps[cut_idx]["t"]},
                "remaining_fields": [f.field_id for f in remaining],
                "n_remaining_fields": len(remaining),
                "skipped_fields": [f.field_id for f in skipped],
                "skipped_area_m2": round(float(sum(f.area_m2 for f in skipped)), 1),
                "strict_nfz": bool(strict),
                "no_replan": {"intrusion_m": round(float(intr_no), 1), "collision": bool(intr_no > 1.0)},
                "replan": {
                    "intrusion_m": round(float(intr_yes), 1),
                    "collision": bool(intr_yes > 1.0),
                    "replan_time_s": round(float(new["metrics"]["planning_time_s"]), 2),
                    "remaining_flight_m": round(float(new["metrics"]["path_length_m"]), 1),
                    "extra_distance_m": round(float(stitched_len - original_len), 1),
                    "total_distance_m": round(float(stitched_len), 1),
                    "energy_total_wh": round(float(_prefix_energy(uav, prefix) + new["metrics"]["energy_wh"]), 2),
                },
                "trajectories": {
                    "original": [[round(float(x), 1), round(float(y), 1)] for x, y in pts],
                    "replanned": [[round(float(w["x"]), 1), round(float(w["y"]), 1)] for w in stitched],
                },
                "error": last_err,
            }
    return None

