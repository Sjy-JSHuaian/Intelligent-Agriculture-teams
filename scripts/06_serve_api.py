# -*- coding: utf-8 -*-
"""交付物5：启动网页接口与地图页面。

用法：
    python scripts/06_serve_api.py              # http://127.0.0.1:8000
    python scripts/06_serve_api.py --port 8010
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    import uvicorn
    print(f"打开 http://{args.host}:{args.port}/  （Ctrl+C 停止）")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
