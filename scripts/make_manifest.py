# -*- coding: utf-8 -*-
"""生成/刷新 manifest.csv（文件、大小、SHA-256、来源、许可）。

用法：python scripts/make_manifest.py
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

FILES = [
    ("Copernicus_DEM_GLO30/raw/Copernicus_DSM_COG_10_N23_00_E102_00_DEM.tif",
     "Copernicus DEM GLO-30 / AWS Open Data", "Copernicus DEM 免费开放（注明来源）"),
    ("OpenStreetMap/raw/osm_yuanyang_core.json",
     "OpenStreetMap contributors / Overpass API", "ODbL 1.0"),
    ("OpenStreetMap/raw/osm_yuanyang_core.geojson",
     "由 osm_yuanyang_core.json 转换", "ODbL 1.0"),
]


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for rel, source, lic in FILES:
        p = BASE / rel
        if not p.exists():
            print(f"跳过（不存在）：{rel}")
            continue
        rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p),
                     "source": source, "license": lic})
    out = BASE / "manifest.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "source", "license"])
        w.writeheader(); w.writerows(rows)
    print(f"写出 {out}：{len(rows)} 个文件")
    for r in rows:
        print(f"  {r['path']}  {r['bytes']/1048576:.1f} MB  {r['sha256'][:16]}…")


if __name__ == "__main__":
    main()
