# -*- coding: utf-8 -*-
"""交付物5：FastAPI 网页接口（只读本地文件，无数据库）。

启动：
    python scripts/06_serve_api.py
    # 或 uvicorn src.api.app:app --reload --port 8000
接口：
    GET /api/scenes
    GET /api/scene/{sid}
    GET /api/traj/{sid}/{method}
    GET /api/inspect/{sid}/{method}
    GET /api/metrics
    GET /api/replan
    静态：/  /static/...  /basemaps/...
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.common.config import EXP_DIR, FRONTEND_DIR, INSPECT_DIR, PLOT_DIR, SCENE_DIR, SCENE_IDS, TRAJ_DIR

app = FastAPI(title="成员A 无人机梯田巡检仿真接口", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
if PLOT_DIR.exists():
    app.mount("/basemaps", StaticFiles(directory=str(PLOT_DIR)), name="basemaps")


def _scene_meta(sid: str) -> Dict:
    p = SCENE_DIR / sid / "scene.json"
    if not p.exists():
        raise HTTPException(404, f"场景不存在：{sid}")
    return json.loads(p.read_text(encoding="utf-8"))


def _metrics_rows() -> List[Dict]:
    p = EXP_DIR / "metrics.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


@app.get("/")
def index():
    p = FRONTEND_DIR / "index.html"
    if not p.exists():
        return JSONResponse({"message": "前端未生成", "docs": "/docs"}, status_code=200)
    return FileResponse(p)


@app.get("/api/scenes")
def list_scenes():
    rows = _metrics_rows()
    out = []
    for sid in SCENE_IDS:
        try:
            meta = _scene_meta(sid)
        except HTTPException:
            continue
        m = {r["method"]: r for r in rows if r["scene"] == sid}
        out.append({
            "scene_id": sid,
            "name": meta["name"],
            "complexity": meta["complexity"],
            "size_m": meta["size_m"],
            "n_fields": len(meta["fields"]),
            "center_lonlat": meta.get("center_lonlat"),
            "relief_m": meta["counts"].get("relief_m"),
            "metrics": m,
        })
    return out


@app.get("/api/scene/{sid}")
def get_scene(sid: str):
    meta = _scene_meta(sid)
    feats = []
    for f in meta["fields"]:
        ring = [[round(float(x), 2), round(float(y), 2)] for x, y in f["polygon"]]
        ring.append(ring[0])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {k: f[k] for k in ("field_id", "risk", "risk_level", "priority", "area_m2", "mean_elev")},
        })
    obstacles = {"type": "FeatureCollection", "features": []}
    op = SCENE_DIR / sid / "obstacles.geojson"
    if op.exists():
        obstacles = json.loads(op.read_text(encoding="utf-8"))
    return {
        "scene_id": sid,
        "name": meta["name"],
        "complexity": meta["complexity"],
        "size_m": meta["size_m"],
        "cell_m": meta["cell_m"],
        "nx": meta["nx"], "ny": meta["ny"],
        "depot_xy": meta["depot_xy"],
        "origin_xy": meta["origin_xy"],
        "crs": meta["crs"],
        "center_lonlat": meta.get("center_lonlat"),
        "bounds": [[0, 0], [meta["size_m"], meta["size_m"]]],
        "basemap_url": f"/basemaps/basemap_{sid}.png",
        "fields": {"type": "FeatureCollection", "features": feats},
        "obstacles": obstacles,
        "counts": meta.get("counts", {}),
        "sources": meta.get("sources", {}),
        "notes": meta.get("notes", []),
    }


@app.get("/api/traj/{sid}/{method}")
def get_traj(sid: str, method: str):
    p = TRAJ_DIR / f"{sid}_{method}.json"
    if not p.exists():
        raise HTTPException(404, "航迹不存在，请先运行 scripts/02_run_planning.py")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/inspect/{sid}/{method}")
def get_inspect(sid: str, method: str):
    p = INSPECT_DIR / f"{sid}_{method}_inspection.json"
    if not p.exists():
        raise HTTPException(404, "巡检结果不存在，请先运行 scripts/03_run_inspection.py")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/metrics")
def get_metrics():
    return _metrics_rows()


@app.get("/api/replan")
def get_replan():
    p = EXP_DIR / "replan.json"
    if not p.exists():
        raise HTTPException(404, "重规划结果不存在，请先运行 scripts/04_run_experiments.py")
    return json.loads(p.read_text(encoding="utf-8"))
