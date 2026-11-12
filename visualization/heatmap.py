"""热力图可视化（用于植被指数等连续值数组）。"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(arr: np.ndarray, out_path: str, title: str | None = None,
                 cmap: str = "RdYlGn", vmin: float | None = None,
                 vmax: float | None = None, dpi: int = 150) -> str:
    """保存热力图为 PNG，返回输出路径。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
