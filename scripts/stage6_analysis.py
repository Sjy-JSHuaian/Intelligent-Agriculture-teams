"""阶段 6 植被/异常分析运行脚本。

对 RGB 影像计算植被指数（VARI/ExG/GLI/NDI）+ 覆盖度 + 低绿度异常 + 裸地，
输出热力图、掩膜与汇总 JSON。

用法：
    python scripts/stage6_analysis.py --input data/raw/loveda/Val --n 3
    python scripts/stage6_analysis.py --input data/raw/jiabang_rgb --n 10

注意：加榜梯田 RGB 影像由 A 同学提供后放 data/raw/jiabang_rgb/，本脚本通用。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.anomaly import detect_bare_land, detect_low_vigor
from analysis.vegetation import analyze_vegetation, vegetation_fraction
from datasets.preprocessing import read_image
from visualization.heatmap import save_heatmap


def _collect_images(input_path: Path, n: int | None) -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    if input_path.is_file():
        return [input_path]
    files = []
    for ext in exts:
        files += glob.glob(str(input_path / "**" / ext), recursive=True)
    files = sorted(set(files))
    return [Path(p) for p in files[:n]]


def _save_mask(mask: np.ndarray, out_path: Path) -> None:
    """保存二值掩膜为 0/255 uint8 PNG（中文路径安全）。"""
    arr = (mask.astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(f"无法编码掩膜: {out_path}")
    buf.tofile(str(out_path))


def _save_overview(image, vari, veg_mask, low_vigor, bare_land, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(image)
    axes[0].set_title("原图")
    axes[1].imshow(vari, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("VARI")
    axes[2].imshow(veg_mask, cmap="Greens")
    axes[2].set_title("植被掩膜")
    axes[3].imshow(low_vigor, cmap="Reds")
    axes[3].set_title("低绿度异常")
    axes[4].imshow(bare_land, cmap="YlOrBr")
    axes[4].set_title("裸地")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/loveda/Val")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(ROOT / args.input, args.n)
    print(f"待分析影像 {len(images)} 张")

    summary = []
    for img_path in images:
        name = img_path.stem
        image = read_image(str(img_path))          # HWC uint8 RGB

        result = analyze_vegetation(image)
        indices = result["indices"]
        vari = indices["VARI"]
        veg_mask = result["vegetation_mask"]
        low_vigor = detect_low_vigor(image)
        bare_land = detect_bare_land(image)

        for idx_name, arr in indices.items():
            vmin, vmax = (-1, 1) if idx_name != "ExG" else (-1, 1)
            save_heatmap(arr, str(out_dir / f"{name}_{idx_name}.png"),
                         title=idx_name, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        _save_mask(veg_mask, out_dir / f"{name}_veg_mask.png")
        _save_mask(low_vigor, out_dir / f"{name}_low_vigor.png")
        _save_mask(bare_land, out_dir / f"{name}_bare_land.png")
        _save_overview(image, vari, veg_mask, low_vigor, bare_land,
                       out_dir / f"{name}_overview.png")

        row = {
            "name": name,
            "vegetation_fraction_gli": vegetation_fraction(image, "GLI"),
            "vegetation_fraction_vari": vegetation_fraction(image, "VARI"),
            "index_statistics": result["index_statistics"],
            "low_vigor_fraction": float(low_vigor.mean()),
            "bare_land_fraction": float(bare_land.mean()),
        }
        summary.append(row)
        print(f"[{name}] 植被覆盖(GLI)={row['vegetation_fraction_gli']:.3f} "
              f"低绿度异常={row['low_vigor_fraction']:.3f} "
              f"裸地={row['bare_land_fraction']:.3f}")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()
