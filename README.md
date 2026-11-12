# 成员A：无人机仿真与自适应路径规划（大创项目）

> 丘陵梯田无人机自主巡检、农情预测与精准变量作业仿真系统 —— 成员A 交付目录。
>
> **本目录所有地块、建筑、树木、电线、风险值均为可复现的仿真数据；DEM 与 OSM 为真实公开数据。**
> 不宣称真实无人机飞行或实测精度。

---

## 1. 交付物 ↔ 代码 ↔ 结果（先看这张表）

| 序号 | 交付物（任务书） | 代码位置 | 一键运行 | 输出 | 最低验收 |
|---|---|---|---|---|---|
| 1 | 梯田仿真场景（3 种复杂度） | `src/simulator/scene_builder.py`、`src/simulator/scene.py` | `python scripts/01_build_scenes.py` | `data/scenes/scene_*/`：`dem.tif`、`field_id.tif`、`fields.shp`、`obstacles.geojson`、`scene.json`、`preview.png` | 3 个场景，均含地块、高程差、障碍物 |
| 2 | 路径规划程序（基线+改进） | `src/planner/`（`astar.py`、`costmap.py`、`coverage.py`、`route.py`） | `python scripts/02_run_planning.py` | `result/traj/{scene}_{method}.json`（固定字段） | 基线与改进均可运行，输入输出格式固定 |
| 3 | 巡检仿真结果 | `src/simulator/inspect.py` | `python scripts/03_run_inspection.py` | `result/inspect/*_inspection.json`、`*_fields.csv`、`images/` | 航迹、地块访问顺序、覆盖率、能耗/距离 |
| 4 | 算法实验 | `scripts/04_run_experiments.py`、`src/simulator/replan.py` | `python scripts/04_run_experiments.py` | `result/experiment/metrics.csv`、`replan.csv`、`result/plot/fig*.png` | 基线 vs 改进：长度、覆盖率、规划时间、安全/碰撞 |
| 5 | 网页接口 | `src/api/app.py`、`frontend/index.html` | `python scripts/06_serve_api.py` | `http://127.0.0.1:8000/`、`/api/*`、`/docs` | 地图底图、航迹坐标、地块 ID、覆盖信息 |
| 附 | 图表 | `scripts/05_make_figures.py` | `python scripts/05_make_figures.py` | `result/plot/fig1~fig5.png`、`basemap_*.png` | 答辩/报告直接用 |

> 脚本编号就是交付顺序。**你只需要按 02 → 03 → 04 → 05 → 06 运行**；
> 01 只在想从原始 DEM 重建场景时运行（场景已随目录提供，可直接用）。

---

## 2. 快速开始（Windows / PowerShell）

```powershell
cd project_A
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts\00_check_env.py          # 自检
python scripts\02_run_planning.py       # 交付物2：两种算法跑 3 个场景，约 30 s
python scripts\03_run_inspection.py     # 交付物3：覆盖率 + 图像索引，约 10 s
python scripts\04_run_experiments.py    # 交付物4：指标表 + 重规划，约 30 s
python scripts\05_make_figures.py       # 图表
python scripts\06_serve_api.py          # 交付物5：浏览器打开 http://127.0.0.1:8000/
```

如果只想看网页，且 `result/` 已存在，可直接运行 `06`。
如果只想重建场景（需要 `rasterio` + `pyshp`）：

```powershell
python scripts\01_build_scenes.py
```

数据在上一级目录：`..\数据集存储\`（DEM 瓦片与 OSM 抽取）。详见 `docs/数据说明.md`。

---

## 3. 目录结构

```
project_A/
├─ data/
│  └─ scenes/
│     ├─ scene_simple/      # 12 块地，700×700 m
│     ├─ scene_medium/      # 22 块地，900×900 m
│     └─ scene_complex/     # 34 块地，1100×1100 m
│        ├─ dem.tif         5 m DEM（EPSG:32648）
│        ├─ slope.tif       坡度
│        ├─ field_id.tif    地块编号栅格
│        ├─ obstacle_mask.tif
│        ├─ grid.npz        算法直接读取的栅格缓存
│        ├─ scene.json      元数据 + 地块列表 + 来源说明
│        ├─ fields.geojson / fields.shp
│        ├─ boundary.geojson / boundary.shp
│        ├─ obstacles.geojson + buildings/trees/roads/ditches/powerlines.shp
│        └─ preview.png
├─ src/
│  ├─ common/              配置、几何、距离变换
│  ├─ simulator/           场景构建与读取、巡检仿真、重规划
│  ├─ planner/             代价地图、A*、覆盖航带、全局路线
│  └─ api/                 FastAPI
├─ scripts/                00~06 按交付物编号
├─ result/
│  ├─ traj/                航迹 JSON（交付物2）
│  ├─ inspect/             巡检 JSON/CSV + 仿真占位图（交付物3）
│  ├─ experiment/          metrics.csv / replan.csv（交付物4）
│  ├─ plot/                图表与网页底图
│  └─ web/                 预留
├─ frontend/               Leaflet 页面（本地 vendor，无需外网）
├─ docs/                   交付物对照表、算法说明、数据说明、接口文档
├─ requirements.txt
└─ README.md
```

---

## 4. 每一步跑出来是什么（当前实测结果）

### 交付物1：场景

```powershell
python scripts\01_build_scenes.py
```

控制台示例：

```
[OK] scene_simple: 12 fields, relief 162.5 m, hard 0.24%
[OK] scene_medium: 22 fields, relief 369.3 m, hard 0.41%
[OK] scene_complex: 34 fields, relief 630.5 m, hard 0.51%
```

生成 3 个场景目录，每个含 5 m DEM、坡度、地块编号、障碍栅格、GeoJSON/Shapefile、
`preview.png`。QGIS 打开 `dem.tif` + `fields.shp` 即可复现论文式场景图。

### 交付物2：路径规划

```powershell
python scripts\02_run_planning.py
```

输出 `result/traj/scene_simple_baseline.json` 等 6 个文件，固定字段：

```json
{
  "schema_version": "1.0",
  "scene": "scene_medium",
  "method": "improved",
  "coords": {"crs": "EPSG:32648", "local_origin_xy": [0,0], "unit": "m"},
  "uav": {"alt_agl_m": 25.0, "footprint_radius_m": 7.88, "gsd_cm": 0.29},
  "route": ["T020", "T014", "T001", "..."],
  "fields": [{"field_id": "T001", "risk": 0.82, "visit_order": 3,
              "entry": [x,y], "exit": [x,y], "coverage_pct": 100.0}],
  "waypoints": [{"seq": 0, "x": ..., "y": ..., "z": ..., "t": 0.0,
                 "heading": 0.0, "field_id": null, "action": "takeoff"}],
  "metrics": {"path_length_m": 6458.7, "energy_wh": 136.46, "coverage_pct": 100.0,
              "planning_time_s": 1.4, "high_risk_mean_order": 4.0, ...}
}
```

控制台示例（改进算法中等场景）：

```
[scene_medium/improved] 22 块地, 航程 6459 m, 时间 2321 s, 能耗 136.5 Wh, 规划 1.4 s
```

### 交付物3：巡检仿真

```powershell
python scripts\03_run_inspection.py
```

每个场景/算法输出：

- `result/inspect/{scene}_{method}_inspection.json`：地块访问顺序、逐地块覆盖率、
  图像索引（`field_id`、`timestamp`、`waypoint_seq`、`x/y/z`、`path`、`simulated:true`）。
- `result/inspect/{scene}_{method}_fields.csv`：逐地块覆盖率表。
- `result/inspect/images/...`：**仿真占位图**（128×128，带 SIMULATED 水印），
  用于打通 A→B 数据链，不是真实田间照片。

控制台示例：

```
[scene_medium/improved] 覆盖率 100.0% (地块均值 100.0%), 图像 192 张, 访问顺序 T018->T002->T015...
```

### 交付物4：算法实验

```powershell
python scripts\04_run_experiments.py
```

输出 `result/experiment/metrics.csv`（主表）与 `replan.csv`（禁飞区重规划）。
当前结果（完整表见 CSV）：

| 场景 | 方法 | 航程 m | 能耗 Wh | 电池占用 | 覆盖率 | 规划 s | 高风险平均次序 |
|---|---|---:|---:|---:|---:|---:|---:|
| 简单 | 基线 | 5021.0 | 79.79 | 26.6% | 100% | 1.4 | 5.75 |
| 简单 | **改进** | **4254.8** | **70.34** | 23.4% | 100% | 1.4 | **2.5** |
| 中等 | 基线 | 9730.9 | 181.62 | 60.5% | 100% | 6.1 | 14.71 |
| 中等 | **改进** | **6458.7** | **136.46** | 45.5% | 100% | 3.8 | **4.0** |
| 复杂 | 基线 | 18501.9 | 314.15 | **104.7%（超出电池）** | 100% | 14.2 | 13.73 |
| 复杂 | **改进** | **11491.7** | **220.64** | 73.5% | 100% | 13.6 | **6.0** |

重规划（改进算法，飞行途中注入临时禁飞区）：

| 场景 | 禁飞区半径 | 原航迹闯入 | 重规划后闯入 | 跳过地块 | 重规划耗时 |
|---|---:|---:|---:|---:|---:|
| 简单 | 45 m | 168.6 m | **0 m** | 1 块 | 0.28 s |
| 中等 | 45 m | 205.0 m | **0 m** | 0 | 1.84 s |
| 复杂 | 40 m | 69.6 m | **0 m** | 0 | 4.37 s |

说明：
- 规划耗时随机器负载波动，**以 `result/experiment/metrics.csv` 为准**；航程/能耗/覆盖率是确定性的。
- 覆盖率三者均为 100%，说明改进算法没有用“少拍”换“少飞”；收益来自排序与转场优化。
- 复杂场景基线耗电 314 Wh > 300 Wh 电池，**单架次不可完成**；改进 220.6 Wh 可完成（保留 15% 余量）。
- 中等场景基线有 1 次碰撞/1 m 安全违规；改进为 0。简单场景障碍稀疏，两者均无碰撞。

### 交付物5：网页接口

```powershell
python scripts\06_serve_api.py
```

浏览器打开 <http://127.0.0.1:8000/>：

- 左侧切换 3 个场景、基线/改进；
- 地图显示地形晕渲底图、地块（按风险着色，点击弹出 `field_id`/风险/覆盖率）、
  航迹、拍照点、道路/建筑/电线；
- 指标卡显示航程、能耗、覆盖率、规划耗时、高风险次序、碰撞/违规；
- 接口：`/api/scenes`、`/api/scene/{sid}`、`/api/traj/{sid}/{method}`、
  `/api/inspect/{sid}/{method}`、`/api/metrics`、`/api/replan`，交互文档 `/docs`。

![网页界面](result/plot/web_dashboard.png)

> 深链接：`http://127.0.0.1:8000/#scene_medium/improved` 可直接打开指定场景/算法。

---

## 5. 算法一句话说明

- **基线**：地块按编号固定顺序；每块地固定方向牛耕式航带；块间用纯距离 A*。
- **改进**：高风险地块优先排序；每块地在 4 种航带方向/入口中选“进场+离场”最短的；
  转场 A* 代价 = 距离 ×（1 + 坡度² + 障碍安全距离 + 道路/沟渠软风险）+ 爬升代价。
- 这是对论文 **mUDAI**（两阶段：先选最佳拍摄位置，再解带约束的路径/VRP）在
  “梯田地块 + 单机 + 地形代价”场景下的落地，不照搬其多机 OR-Tools 求解器。
- 公式、参数、复现种子见 `docs/算法说明.md`。

---

## 6. 坐标与接口约定（给 B、C）

- 坐标系：场景局部平面，原点为场景西南角，**x 向东、y 向北，单位 m**；
  GeoTIFF/Shapefile 为 **EPSG:32648**。
- 地块编号：`T001`…`T0NN`，三个场景独立编号；`scene.json` 与所有 JSON 一致。
- 时间戳：ISO 8601，带 `+08:00`；巡检起点固定 `2026-10-01T08:00:00+08:00`。
- 风险字段：`risk`（0–1）、`risk_level`（low/medium/high）、`priority`（1–3）。
- 给 B 的图像索引：`result/inspect/{scene}_{method}_inspection.json` 的 `images[]`；
  给 C 的风险反馈接口：`{field_id, timestamp, risk_level, predicted_risk, inspection_priority}`。

详见 `docs/接口文档.md`。

---

## 7. 复现性与边界

- 场景随机种子：11 / 22 / 33；Python `numpy` 固定种子，结果可完全复现。
- 所有图由 `result/` 下的 CSV/JSON 生成，不手填数字。
- **真实**：Copernicus DEM GLO-30、OpenStreetMap 道路/水系/林地。
- **仿真**：地块边界、田埂、建筑、树木、电线、病害风险值、巡检照片（占位图）。
- 不包含：真实无人机飞行、真实病害识别模型、真实产量/施药量。

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: rasterio` | 只影响 `01_build_scenes.py`；场景已生成，可直接跑 02–06。需要重建时 `pip install rasterio pyshp` |
| `A*: 未找到路径` | 场景被新障碍堵死；删除 `data/scenes/scene_*` 后重跑 01，或换种子 |
| 网页空白 | 先跑 02/03 生成 traj/inspect；确认 8000 端口未被占用；F12 看 `/api/*` 是否 404 |
| 中文图乱码 | 已设置 Microsoft YaHei；若系统无该字体，安装或改 `05_make_figures.py` 顶部字体 |
| 覆盖率不是 100% | 检查是否改了 `UAVSpec.fov_deg` 或 `overlap`；默认参数下应≥99% |

---

## 9. 主要图表

![三套场景](result/plot/fig1_scenes.png)

![航迹对比](result/plot/fig2_traj_compare.png)

![指标对比](result/plot/fig3_metrics.png)

![覆盖](result/plot/fig4_coverage.png)

![重规划](result/plot/fig5_replan.png)
