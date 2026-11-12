# -*- coding: utf-8 -*-
"""成员A 全局配置：路径、场景清单、无人机/相机参数、代价权重。

所有路径均相对于 project_A/，不写死盘符，换电脑后可直接运行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------- 目录
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SCENE_DIR = DATA_DIR / "scenes"
RESULT_DIR = ROOT / "result"
TRAJ_DIR = RESULT_DIR / "traj"
INSPECT_DIR = RESULT_DIR / "inspect"
PLOT_DIR = RESULT_DIR / "plot"
EXP_DIR = RESULT_DIR / "experiment"
WEB_DIR = RESULT_DIR / "web"
FRONTEND_DIR = ROOT / "frontend"
DOCS_DIR = ROOT / "docs"
DATASET_DIR = ROOT.parent / "数据集存储"

for _d in (SCENE_DIR, TRAJ_DIR, INSPECT_DIR, PLOT_DIR, EXP_DIR, WEB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "1.0"
CRS_UTM = "EPSG:32648"          # WGS84 / UTM zone 48N（元阳经度约 102.7°E）
CRS_WGS84 = "EPSG:4326"


# ---------------------------------------------------------------- 场景
@dataclass(frozen=True)
class SceneSpec:
    """一套梯田仿真场景的定义。

    中心点位于云南元阳哈尼梯田核心区；DEM 为真实 Copernicus GLO-30，
    地块/障碍物为在真实地形上人工布设的仿真要素（见 docs/数据说明.md）。
    """
    sid: str
    name: str
    complexity: str
    center_lon: float
    center_lat: float
    size_m: float
    n_fields: int
    n_trees: int
    n_powerlines: int
    seed: int
    obstacle_level: float = 0.5

    @property
    def nx(self) -> int:
        return int(round(self.size_m / CELL_M))

    @property
    def ny(self) -> int:
        return int(round(self.size_m / CELL_M))


CELL_M = 5.0                     # 规划栅格分辨率（m），对应“5 m 局部仿真网格”
RISK_LEVELS = ("low", "medium", "high")

SCENE_SPECS: Dict[str, SceneSpec] = {
    "scene_simple": SceneSpec(
        sid="scene_simple", name="简单场景·箐口缓坡梯田", complexity="simple",
        center_lon=102.7481, center_lat=23.1169, size_m=700,
        n_fields=12, n_trees=14, n_powerlines=0, seed=11, obstacle_level=0.35,
    ),
    "scene_medium": SceneSpec(
        sid="scene_medium", name="中等场景·多依树密集梯田", complexity="medium",
        center_lon=102.7625, center_lat=23.1167, size_m=900,
        n_fields=22, n_trees=28, n_powerlines=1, seed=22, obstacle_level=0.6,
    ),
    "scene_complex": SceneSpec(
        sid="scene_complex", name="复杂场景·老虎嘴陡坡梯田", complexity="complex",
        center_lon=102.7418, center_lat=23.0821, size_m=1100,
        n_fields=34, n_trees=46, n_powerlines=2, seed=33, obstacle_level=0.85,
    ),
}

SCENE_IDS: List[str] = list(SCENE_SPECS.keys())


# ---------------------------------------------------------------- 无人机 / 相机
@dataclass(frozen=True)
class UAVSpec:
    """无人机平台与下视相机参数（仿真标定值，非真实机型实测）。"""
    cruise_speed: float = 6.0        # m/s 水平巡航速度
    climb_speed: float = 2.0         # m/s 爬升/下降速度
    alt_agl: float = 25.0            # m 相对地面飞行高度
    alt_min: float = 15.0
    alt_max: float = 60.0
    fov_deg: float = 35.0            # 下视相机视场角
    img_w: int = 5472
    img_h: int = 3648
    battery_wh: float = 300.0
    reserve_ratio: float = 0.15      # 低于 15% 电量必须返航
    dwell_s: float = 2.0             # 每个拍照点悬停时间
    p_cruise_w: float = 180.0
    p_climb_w: float = 260.0
    p_hover_w: float = 210.0
    p_avionics_w: float = 25.0
    mass_kg: float = 4.5
    takeoff_s: float = 8.0
    landing_s: float = 8.0

    @property
    def footprint_radius(self) -> float:
        return self.alt_agl * math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def footprint_width(self) -> float:
        return 2.0 * self.footprint_radius

    def swath(self, overlap: float = 0.15) -> float:
        """航带间距：相邻航带按 overlap 重叠。"""
        return max(CELL_M, self.footprint_width * (1.0 - overlap))

    def capture_step(self, overlap: float = 0.15) -> float:
        """沿航向拍照间距。"""
        return max(CELL_M, self.footprint_width * (1.0 - overlap) * 0.65)

    @property
    def gsd_cm(self) -> float:
        """地面分辨率（cm/px），按画幅宽度与像元数估算。"""
        return self.footprint_width * 100.0 / self.img_w


DEFAULT_UAV = UAVSpec()


# ---------------------------------------------------------------- 代价权重
@dataclass(frozen=True)
class CostWeights:
    """改进 A* 的代价权重；基线只用几何距离。"""
    w_slope: float = 2.0            # 坡度惩罚
    slope_ref_deg: float = 20.0
    w_clear: float = 6.0            # 障碍物安全距离惩罚
    d_safe: float = 8.0             # m 进入该范围开始惩罚
    w_soft: float = 0.35            # 道路/沟渠等软风险
    w_climb: float = 0.8            # 爬升代价（每米）
    slope_max_deg: float = 38.0     # 超过视为不可通行
    block_clear_m: float = 2.0      # 与障碍物距离小于该值视为禁入
    violation_clear_m: float = 3.0  # 统计“安全违规”的阈值


DEFAULT_WEIGHTS = CostWeights()


# ---------------------------------------------------------------- 规划参数
@dataclass(frozen=True)
class PlanParams:
    start_xy: Tuple[float, float] = (40.0, 40.0)   # 默认起降点（若被占则就近吸附）
    overlap: float = 0.15
    risk_weight: float = 3.0         # 加权完成时间中的风险权重
    rng_seed: int = 2026
    astar_max_expand: int = 400_000


DEFAULT_PLAN = PlanParams()

METHODS = ("baseline", "improved")
METHOD_NAMES = {"baseline": "基线：固定顺序 + 纯距离 A*", "improved": "改进：风险优先 + 地形/安全/能耗代价 A*"}

