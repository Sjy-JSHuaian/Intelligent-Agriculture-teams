# -*- coding: utf-8 -*-
"""交付物3/4 的图表与网页底图：场景、航迹对比、指标、覆盖、重规划。

用法：python scripts/05_make_figures.py
输出：result/plot/*.png 与 result/plot/basemap_*.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Circle, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import EXP_DIR, INSPECT_DIR, PLOT_DIR, SCENE_IDS, TRAJ_DIR
from src.simulator.scene import Scene

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

RISK_COLOR = {"low": "#43a047", "medium": "#fb8c00", "high": "#e53935"}
METHOD_COLOR = {"baseline": "#546e7a", "improved": "#e53935"}
METHOD_LABEL = {"baseline": "基线（固定顺序 + 纯距离 A*）", "improved": "改进（风险优先 + 地形/安全/能耗代价）"}


def hillshade(scene: Scene):
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(scene.elevation, vert_exag=1.6, dx=scene.cell, dy=scene.cell)


def load_all():
    scenes = {sid: Scene.load(sid) for sid in SCENE_IDS}
    trajs = {}
    for sid in SCENE_IDS:
        for m in ("baseline", "improved"):
            p = TRAJ_DIR / f"{sid}_{m}.json"
            if p.exists():
                trajs[(sid, m)] = json.loads(p.read_text(encoding="utf-8"))
    metrics = []
    mp = EXP_DIR / "metrics.csv"
    if mp.exists():
        metrics = list(csv.DictReader(mp.open(encoding="utf-8-sig")))
    replan = {}
    rp = EXP_DIR / "replan.json"
    if rp.exists():
        replan = json.loads(rp.read_text(encoding="utf-8"))
    return scenes, trajs, metrics, replan


def draw_scene(ax, scene, show_labels=True, title=True):
    hs = hillshade(scene)
    ax.imshow(hs, cmap="gray", origin="upper", extent=[0, scene.size_m, 0, scene.size_m], alpha=0.9)
    # 等高线增强梯田感
    step = 20 if scene.size_m > 1000 else 10
    levels = np.arange(np.floor(scene.elevation.min() / step) * step, scene.elevation.max(), step)
    ax.contour(np.linspace(0, scene.size_m, scene.nx), np.linspace(0, scene.size_m, scene.ny),
               scene.elevation, levels=levels, colors="black", linewidths=0.25, alpha=0.25)
    for f in scene.fields:
        p = f.polygon
        ax.fill(p[:, 0], p[:, 1], color=RISK_COLOR[f.risk_level], alpha=0.70,
                edgecolor="white", linewidth=0.6, zorder=3)
        if show_labels:
            ax.text(f.centroid[0], f.centroid[1], f.field_id, fontsize=4.2, ha="center",
                    va="center", color="white", zorder=5)
    # 障碍物（硬）
    hr, hc = np.where(scene.hard)
    if len(hr):
        ax.scatter(hc * scene.cell, scene.size_m - hr * scene.cell, s=0.3, c="#263238",
                   alpha=0.55, linewidths=0, zorder=4)
    # 道路
    for g in scene.meta.get("sources", {}).get("_", []):
        pass
    ax.plot(scene.depot[0], scene.depot[1], marker="*", ms=15, color="#00e5ff",
            markeredgecolor="black", zorder=8)
    ax.set_xlim(0, scene.size_m); ax.set_ylim(0, scene.size_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m（东）", fontsize=8); ax.set_ylabel("y / m（北）", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(f"{scene.name}\n{scene.size_m:.0f}×{scene.size_m:.0f} m · {len(scene.fields)} 块地",
                     fontsize=10)


def fig_scenes(scenes):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    for ax, sid in zip(axes, SCENE_IDS):
        draw_scene(ax, scenes[sid])
    handles = [Patch(color=RISK_COLOR[k], label={"low": "低风险", "medium": "中风险", "high": "高风险"}[k])
               for k in ("low", "medium", "high")]
    handles += [plt.Line2D([], [], marker="*", color="#00e5ff", markeredgecolor="black", ls="", ms=12, label="起降点"),
                plt.Line2D([], [], marker=".", color="#263238", ls="", ms=6, label="建筑/树木/电线")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("三套梯田仿真场景（真实 GLO-30 地形底图 + 人工地块/障碍物）", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = PLOT_DIR / "fig1_scenes.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def fig_traj(scenes, trajs):
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    for i, sid in enumerate(SCENE_IDS):
        sc = scenes[sid]
        for j, m in enumerate(("baseline", "improved")):
            ax = axes[i, j]
            draw_scene(ax, sc, show_labels=False, title=False)
            tr = trajs.get((sid, m))
            if tr:
                pts = np.array([[w["x"], w["y"]] for w in tr["waypoints"]])
                ax.plot(pts[:, 0], pts[:, 1], color=METHOD_COLOR[m], lw=1.1, alpha=0.9, zorder=6)
                cap = np.array([[w["x"], w["y"]] for w in tr["waypoints"] if w["action"] == "capture"])
                if len(cap):
                    ax.scatter(cap[:, 0], cap[:, 1], s=2.0, color="#212121", alpha=0.5, zorder=7)
                mm = tr["metrics"]
                ax.set_title(f"{'基线' if m == 'baseline' else '改进'}：{mm['path_length_m']:.0f} m / "
                             f"{mm['energy_wh']:.1f} Wh / 高风险次序 {mm['high_risk_mean_order']:.1f}",
                             fontsize=9)
            ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("三套场景下基线与改进算法航迹（红=改进，蓝灰=基线；黑点=拍照点）", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOT_DIR / "fig2_traj_compare.png"; fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("saved", out)

def fig_metrics(metrics):
    if not metrics:
        print("skip fig3: no metrics.csv")
        return
    by = {(r["scene"], r["method"]): r for r in metrics}
    scenes = [s for s in SCENE_IDS if (s, "baseline") in by]
    labels = {"scene_simple": "简单", "scene_medium": "中等", "scene_complex": "复杂"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("path_length_m", "航程 / m", 1.0),
        ("energy_wh", "模拟能耗 / Wh", 1.0),
        ("high_risk_mean_order", "高风险地块平均访问次序（越小越早）", 1.0),
        ("planning_time_s", "规划耗时 / s", 1.0),
    ]
    for ax, (key, title, scale) in zip(axes.ravel(), specs):
        x = np.arange(len(scenes)); w = 0.36
        b = [float(by[(s, "baseline")][key]) * scale for s in scenes]
        m = [float(by[(s, "improved")][key]) * scale for s in scenes]
        ax.bar(x - w / 2, b, w, color=METHOD_COLOR["baseline"], label="基线")
        ax.bar(x + w / 2, m, w, color=METHOD_COLOR["improved"], label="改进")
        ax.set_xticks(x); ax.set_xticklabels([labels.get(s, s) for s in scenes])
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25)
        for xi, v in zip(x - w / 2, b):
            ax.text(xi, v, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        for xi, v in zip(x + w / 2, m):
            ax.text(xi, v, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("基线与改进算法定量对比", fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / "fig3_metrics.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def fig_coverage(scenes, trajs):
    sid = "scene_medium"
    sc = scenes[sid]
    tr = trajs.get((sid, "improved"))
    if not tr:
        return
    insp = INSPECT_DIR / f"{sid}_improved_inspection.json"
    if not insp.exists():
        return
    data = json.loads(insp.read_text(encoding="utf-8"))
    r = sc.uav().footprint_radius

    # 选取 4 块彼此靠近的地块做局部放大，展示航带与拍照幅面
    centers = np.array([f.centroid for f in sc.fields])
    target = np.median(centers, axis=0)
    near = sorted(sc.fields, key=lambda f: float(np.linalg.norm(f.centroid - target)))[:4]
    xs = np.concatenate([f.polygon[:, 0] for f in near]); ys = np.concatenate([f.polygon[:, 1] for f in near])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(160, (xs.max() - xs.min()) / 2 + 70, (ys.max() - ys.min()) / 2 + 70)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(hillshade(sc), cmap="gray", extent=[0, sc.size_m, 0, sc.size_m], origin="upper")
    for f in near:
        p = f.polygon
        ax.fill(p[:, 0], p[:, 1], color=RISK_COLOR[f.risk_level], alpha=0.30, edgecolor="white", lw=0.8, zorder=3)
        ax.text(f.centroid[0], f.centroid[1], f"{f.field_id}\n{f.risk_level}", fontsize=7, color="white",
                ha="center", va="center", zorder=6)
    # 拍照幅面（圆形近似）
    caps = [w for w in tr["waypoints"] if w["action"] == "capture" and
            abs(w["x"] - cx) < half + 50 and abs(w["y"] - cy) < half + 50]
    for w in caps:
        ax.add_patch(Circle((w["x"], w["y"]), r, color="#4fc3f7", alpha=0.18, lw=0.4, ec="#0288d1", zorder=4))
        ax.plot(w["x"], w["y"], ".", color="#ffeb3b", ms=3, zorder=5)
    pts = np.array([[w["x"], w["y"]] for w in tr["waypoints"]])
    ax.plot(pts[:, 0], pts[:, 1], color="#e53935", lw=1.0, alpha=0.7, zorder=5)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(f"覆盖航带与拍照幅面（{sc.name}，改进算法）\n"
                 f"整场覆盖率 {data['totals']['coverage_pct']:.1f}% · {data['totals']['n_images']} 张照片 · "
                 f"单张幅面约 {2*r:.1f} m", fontsize=10)
    ax.set_xlabel("x / m（东）"); ax.set_ylabel("y / m（北）")
    fig.tight_layout()
    out = PLOT_DIR / "fig4_coverage.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)

def fig_replan(scenes, replan):
    if not replan:
        return
    n = len(replan)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    for ax, (sid, r) in zip(axes[0], replan.items()):
        sc = scenes[sid]
        draw_scene(ax, sc, show_labels=False, title=False)
        orig = np.array(r["trajectories"]["original"])
        new = np.array(r["trajectories"]["replanned"])
        ax.plot(orig[:, 0], orig[:, 1], "--", color="#546e7a", lw=1.1, label="原航迹")
        ax.plot(new[:, 0], new[:, 1], "-", color="#e53935", lw=1.4, label="重规划航迹")
        c = r["nfz"]["center"]
        ax.add_patch(Circle(c, r["nfz"]["radius_m"], color="#d50000", alpha=0.22, zorder=2))
        ax.plot(c[0], c[1], "x", color="#b71c1c", ms=8, zorder=9)
        ax.set_title(f"{sc.name}\n禁飞区闯入 {r['no_replan']['intrusion_m']:.0f} m → "
                     f"{r['replan']['intrusion_m']:.0f} m", fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("临时禁飞区注入与局部重规划", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    out = PLOT_DIR / "fig5_replan.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


def basemaps(scenes):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    terra = LinearSegmentedColormap.from_list(
        "terra", ["#223528", "#3c5634", "#687443", "#9a875d", "#c4b79c", "#e2ddd0"])
    for sid, sc in scenes.items():
        hs = hillshade(sc)
        rgb = terra(Normalize()(sc.elevation))[..., :3]
        shade = 0.40 + 0.70 * hs
        img = np.clip(rgb * shade[..., None] * 1.05, 0, 1)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, origin="upper", extent=[0, sc.size_m, 0, sc.size_m])
        ax.set_xlim(0, sc.size_m); ax.set_ylim(0, sc.size_m); ax.axis("off")
        out = PLOT_DIR / f"basemap_{sid}.png"
        fig.savefig(out, dpi=130, pad_inches=0)
        plt.close(fig)
        print("saved", out)


def main():
    scenes, trajs, metrics, replan = load_all()
    fig_scenes(scenes)
    fig_traj(scenes, trajs)
    fig_metrics(metrics)
    fig_coverage(scenes, trajs)
    fig_replan(scenes, replan)
    basemaps(scenes)


if __name__ == "__main__":
    main()




