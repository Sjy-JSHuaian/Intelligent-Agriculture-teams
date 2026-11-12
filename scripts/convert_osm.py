# -*- coding: utf-8 -*-
"""把 Overpass API 的 JSON 转成 QGIS 可直接打开的 GeoJSON。

用法：
    python scripts/convert_osm.py
输出：
    OpenStreetMap/raw/osm_yuanyang_core.geojson
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "OpenStreetMap" / "raw" / "osm_yuanyang_core.json"
DST = BASE / "OpenStreetMap" / "raw" / "osm_yuanyang_core.geojson"

AREA_TAGS = ("building", "landuse", "natural", "amenity", "leisure")


def kind(tags: dict) -> str:
    if "building" in tags:
        return "building"
    if "highway" in tags:
        return "road"
    if "waterway" in tags:
        return "waterway"
    if tags.get("natural") in ("wood", "scrub"):
        return "forest"
    if "landuse" in tags:
        return "landuse"
    if "power" in tags:
        return "powerline"
    return "other"


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    feats = []
    for e in data.get("elements", []):
        if e.get("type") != "way" or "geometry" not in e:
            continue
        tags = e.get("tags", {}) or {}
        coords = [[round(p["lon"], 7), round(p["lat"], 7)] for p in e["geometry"]]
        if len(coords) < 2:
            continue
        closed = len(coords) >= 4 and coords[0] == coords[-1]
        is_area = closed and any(t in tags for t in AREA_TAGS)
        geom = {"type": "Polygon", "coordinates": [coords]} if is_area else {"type": "LineString", "coordinates": coords}
        feats.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "osm_id": e.get("id"),
                "kind": kind(tags),
                "name": tags.get("name", ""),
                "highway": tags.get("highway", ""),
                "waterway": tags.get("waterway", ""),
                "landuse": tags.get("landuse", ""),
                "natural": tags.get("natural", ""),
                "building": tags.get("building", ""),
                "power": tags.get("power", ""),
            },
        })
    fc = {"type": "FeatureCollection", "features": feats}
    DST.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(f"写出 {DST}：{len(feats)} 个要素")


if __name__ == "__main__":
    main()
