# -*- coding: utf-8 -*-
"""一键下载成员A所需公开数据（Copernicus DEM GLO-30 + OpenStreetMap）。

用法（在本目录下）：
    python scripts/download_all.py              # DEM + OSM
    python scripts/download_all.py --dem-only
    python scripts/download_all.py --osm-only

说明：
- DEM 使用 AWS Open Data 的 Copernicus GLO-30 公开镜像，无需账号。
- OSM 使用 Overpass API 按研究区 bbox 精确抽取（Geofabrik 中国全量 PBF 约 1.4 GB，
  只做元阳小区域时没有必要；如确需全量请到 https://download.geofabrik.de/asia/china.html）。
- Google Earth Engine 的 DEM 目录需要 Google 账号授权，不在本脚本自动下载，
  代码见 GEE/export_dem_gee.js。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEM_DIR = BASE / "Copernicus_DEM_GLO30" / "raw"
OSM_DIR = BASE / "OpenStreetMap" / "raw"

DEM_TILE = "Copernicus_DSM_COG_10_N23_00_E102_00_DEM"
DEM_URL = f"https://copernicus-dem-30m.s3.amazonaws.com/{DEM_TILE}/{DEM_TILE}.tif"
# 研究区：云南元阳哈尼梯田核心区（覆盖 3 套场景）
BBOX = (23.030, 102.690, 23.132, 102.782)   # south, west, north, east
OSM_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

OSM_QUERY = f"""[out:json][timeout:180];
(
  way["highway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["waterway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["building"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["landuse"~"farmland|orchard|forest|meadow"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["natural"~"wood|scrub"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["power"~"line|minor_line"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out geom;"""


def download(url: str, dst: Path, timeout: int = 120):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "dachuang-memberA/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            b = r.read(1024 * 256)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total and done % (8 * 1024 * 1024) < 256 * 1024:
                print(f"  {done/1048576:.1f}/{total/1048576:.1f} MB", flush=True)
    os.replace(tmp, dst)
    print(f"  完成：{dst}（{dst.stat().st_size/1048576:.1f} MB，{time.time()-t0:.1f} s）")


def fetch_osm(dst: Path):
    body = urllib.parse.urlencode({"data": OSM_QUERY}).encode()
    last = None
    for url in OSM_URLS:
        for attempt in range(2):
            try:
                print(f"  请求 {url}（第 {attempt+1} 次）…")
                req = urllib.request.Request(url, data=body, headers={
                    "User-Agent": "dachuang-memberA/1.0 (research; OSM Overpass)"})
                raw = urllib.request.urlopen(req, timeout=240).read()
                data = json.loads(raw)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(raw)
                print(f"  完成：{dst}（{len(data.get('elements', []))} 个要素，{len(raw)/1024:.0f} KB）")
                return True
            except Exception as e:
                last = e
                print(f"  失败：{type(e).__name__}: {str(e)[:120]}")
                time.sleep(3)
    print(f"  OSM 下载失败（{last}）。可改用浏览器打开 Overpass Turbo 导出。")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem-only", action="store_true")
    ap.add_argument("--osm-only", action="store_true")
    args = ap.parse_args()

    if not args.osm_only:
        print("[1/2] Copernicus DEM GLO-30")
        dst = DEM_DIR / f"{DEM_TILE}.tif"
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"  已存在，跳过：{dst}")
        else:
            download(DEM_URL, dst)

    if not args.dem_only:
        print("[2/2] OpenStreetMap（Overpass API）")
        dst = OSM_DIR / "osm_yuanyang_core.json"
        if dst.exists() and dst.stat().st_size > 10_000:
            print(f"  已存在，跳过：{dst}（如需更新请先删除）")
        else:
            fetch_osm(dst)

    # 顺手转一份 GeoJSON 方便 QGIS
    try:
        sys.path.insert(0, str(BASE / "scripts"))
        import convert_osm  # noqa: F401
        convert_osm.main()
    except Exception as e:
        print(f"  GeoJSON 转换跳过：{e}")

    print("\n数据目录：", BASE)
    print("下一步：按 README.md 在 QGIS 中查看，或在 project_A 中重建场景。")


if __name__ == "__main__":
    main()
