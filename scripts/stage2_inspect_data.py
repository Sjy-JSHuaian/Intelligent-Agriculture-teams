"""阶段 2：数据读取、统计与可视化检查脚本。

用法（在 image_analysis 根目录运行）：
    python scripts/stage2_inspect_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.dataset_loader import collect_samples, compute_statistics, print_statistics  # noqa: E402
from datasets.preprocessing import compute_rgb_indices, read_image, read_mask, to_float01  # noqa: E402
from visualization.heatmap import save_heatmap  # noqa: E402
from visualization.segmentation_visualizer import save_color_mask, save_overlay  # noqa: E402

DATA_ROOT = ROOT / "data" / "raw" / "loveda"
FIG_DIR = ROOT / "outputs" / "figures"


def main() -> None:
    if not DATA_ROOT.exists():
        print(f"[错误] 数据目录不存在: {DATA_ROOT}")
        print("请先将 LoveDA 的 Train/Val/Test 解压到 data/raw/loveda/")
        sys.exit(1)

    # 1) 数据集统计
    for split in ["train", "val"]:
        try:
            stats = compute_statistics(str(DATA_ROOT), split)
            print_statistics(stats)
        except FileNotFoundError as e:
            print(f"[跳过] {e}")

    # 2) 样例可视化（原始图 + 叠加图 + 彩色掩膜 + RGB 植被指数）
    pairs = collect_samples(str(DATA_ROOT), "train")
    if not pairs:
        print("[警告] train 无样本，跳过可视化")
        return

    for img_path, mask_path in pairs[:3]:
        image = read_image(img_path)
        mask = read_mask(mask_path)
        name = Path(img_path).stem
        save_overlay(image, mask, str(FIG_DIR / f"overlay_{name}.png"))
        save_color_mask(mask, str(FIG_DIR / f"mask_{name}.png"))
        indices = compute_rgb_indices(to_float01(image))
        for key, arr in indices.items():
            save_heatmap(arr, str(FIG_DIR / f"{key}_{name}.png"), title=f"{key} (RGB)")
        print(f"[可视化] 已生成样例 {name} 的叠加图 / 掩膜 / 植被指数图")

    print("[完成] 阶段 2 数据检查结束")


if __name__ == "__main__":
    main()
