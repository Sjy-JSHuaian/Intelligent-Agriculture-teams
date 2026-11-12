# -*- coding: utf-8 -*-
"""巡检仿真：沿航迹生成拍照点覆盖栅格、地块访问记录与图像索引。

说明：图像索引中的 jpg 是由地形/风险纹理生成的“仿真占位图”，
仅用于打通 A→B 的数据链，不是真实田间照片，文件内含 SIMULATED 水印。
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.common.config import INSPECT_DIR, SCHEMA_VERSION
from src.simulator.scene import Scene

TZ = timezone(timedelta(hours=8))
MISSION_START = datetime(2026, 10, 1, 8, 0, 0, tzinfo=TZ)


def _draw_sim_image(path: Path, field_id: str, risk: float, seed: int, stamp: str) -> None:
    """生成 128x128 仿真占位图（非真实照片）。"""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    base = np.zeros((128, 128, 3), dtype=np.float32)
    base[..., 0] = 70 + 60 * risk
    base[..., 1] = 120 + 40 * (1 - risk)
    base[..., 2] = 45 + 25 * (1 - risk)
    noise = rng.normal(0, 18, (128, 128, 1))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(img).resize((128, 128))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 127, 12], fill=(0, 0, 0))
    d.text((2, 1), "SIMULATED", fill=(255, 255, 255))
    d.rectangle([0, 112, 127, 127], fill=(0, 0, 0))
    d.text((2, 114), f"{field_id} {stamp[11:19]}", fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=82)


def inspect_mission(
    traj: Dict,
    scene: Scene,
    generate_images: bool = True,
    out_dir: Path = INSPECT_DIR,
    start: datetime = MISSION_START,
) -> Dict:
    uav = scene.uav()
    r = uav.footprint_radius
    covered = np.zeros((scene.ny, scene.nx), dtype=bool)

    captures = [w for w in traj["waypoints"] if w["action"] == "capture"]
    for w in captures:
        rr, cc = scene.nearest_index((w["x"], w["y"]))
        rad = int(math.ceil(r / scene.cell))
        r0, r1 = max(0, rr - rad), min(scene.ny, rr + rad + 1)
        c0, c1 = max(0, cc - rad), min(scene.nx, cc + rad + 1)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        xs = scene.origin_xy[0] + (xx + 0.5) * scene.cell
        ys = scene.origin_xy[1] + scene.size_m - (yy + 0.5) * scene.cell
        mask = (xs - w["x"]) ** 2 + (ys - w["y"]) ** 2 <= r * r
        covered[r0:r1, c0:c1] |= mask

    field_rows = []
    total_cells = 0
    total_cov = 0
    for rec in traj["fields"]:
        fid = rec["field_id"]
        fm = scene.field_mask(fid)
        n = int(fm.sum())
        cov = int((fm & covered).sum())
        pct = 100.0 * cov / max(n, 1)
        total_cells += n
        total_cov += cov
        field_rows.append({
            "field_id": fid,
            "risk": rec["risk"],
            "risk_level": rec["risk_level"],
            "visit_order": rec["visit_order"],
            "area_m2": round(float(n * scene.cell * scene.cell), 1),
            "covered_m2": round(float(cov * scene.cell * scene.cell), 1),
            "coverage_pct": round(pct, 2),
            "n_images": 0,
        })
    fmap = {r["field_id"]: r for r in field_rows}

    images = []
    for k, w in enumerate(captures):
        fid = w["field_id"]
        ts = start + timedelta(seconds=float(w["t"]))
        seq = fmap[fid]["n_images"] + 1
        fmap[fid]["n_images"] = seq
        rel = Path("images") / scene.scene_id / traj["method"] / f"{fid}_{seq:03d}.jpg"
        if generate_images:
            _draw_sim_image(out_dir / rel, fid, float(fmap[fid]["risk"]), seed=1000 + k, stamp=ts.isoformat())
        images.append({
            "image_id": f"{scene.scene_id}_{traj['method']}_{k:04d}",
            "field_id": fid,
            "timestamp": ts.isoformat(timespec="seconds"),
            "waypoint_seq": w["seq"],
            "x": w["x"], "y": w["y"], "z": w["z"],
            "heading": w.get("heading", 0.0),
            "gsd_cm": round(uav.gsd_cm, 2),
            "path": str(rel).replace("\\", "/"),
            "simulated": True,
            "note": "仿真占位图，非真实田间照片；成员B可用真实/渲染图替换同名字段",
        })

    total_pct = 100.0 * total_cov / max(total_cells, 1)
    mean_field_pct = float(np.mean([f["coverage_pct"] for f in field_rows])) if field_rows else 0.0

    result = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene.scene_id,
        "method": traj["method"],
        "created_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "mission_start": start.isoformat(timespec="seconds"),
        "visit_order": [f["field_id"] for f in field_rows],
        "fields": field_rows,
        "totals": {
            "n_fields": len(field_rows),
            "n_images": len(images),
            "coverage_pct": round(total_pct, 2),
            "mean_field_coverage_pct": round(mean_field_pct, 2),
            "covered_area_m2": round(float(total_cov * scene.cell * scene.cell), 1),
            "field_area_m2": round(float(total_cells * scene.cell * scene.cell), 1),
        },
        "images": images,
        "coverage_grid": {
            "cell_m": scene.cell, "nx": scene.nx, "ny": scene.ny,
            "origin_xy": scene.utm_origin.tolist(),
            "note": "coverage.tif 与场景 dem.tif 对齐，1=被至少一张照片覆盖",
        },
    }

    # 写覆盖栅格（QGIS 可打开）
    try:
        import rasterio
        from rasterio.transform import from_origin
        tif = out_dir / f"{scene.scene_id}_{traj['method']}_coverage.tif"
        with rasterio.open(
            tif, "w", driver="GTiff", height=scene.ny, width=scene.nx, count=1,
            dtype="uint8", crs="EPSG:32648",
            transform=from_origin(scene.utm_origin[0], scene.utm_origin[1] + scene.size_m, scene.cell, scene.cell),
            compress="lzw",
        ) as ds:
            ds.write(covered.astype(np.uint8), 1)
    except Exception as e:  # pragma: no cover
        result["coverage_grid"]["error"] = f"写 GeoTIFF 失败：{e}"

    return result


def save_inspection(result: Dict, traj: Dict, out_dir: Path = INSPECT_DIR) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{result['scene']}_{result['method']}_inspection.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    cp = out_dir / f"{result['scene']}_{result['method']}_fields.csv"
    with open(cp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["fields"][0].keys()))
        w.writeheader(); w.writerows(result["fields"])

    # 把覆盖率回写进航迹 JSON，保证单个文件自洽
    traj["metrics"]["coverage_pct"] = result["totals"]["coverage_pct"]
    traj["metrics"]["coverage_pct_mean_field"] = result["totals"]["mean_field_coverage_pct"]
    traj["metrics"]["n_images"] = result["totals"]["n_images"]
    cov = {f["field_id"]: f["coverage_pct"] for f in result["fields"]}
    for rec in traj["fields"]:
        rec["coverage_pct"] = cov.get(rec["field_id"])
    tp = Path(traj.get("_path", "")) if traj.get("_path") else None
    if tp and tp.exists():
        with open(tp, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in traj.items() if not k.startswith("_")}, f, ensure_ascii=False, indent=1)
    return {"inspection_json": str(jp), "fields_csv": str(cp)}
