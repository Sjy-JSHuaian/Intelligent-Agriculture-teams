# 加榜梯田 C 同学：气象风险分析与协同模块

这个包是可运行的Python原型：用已经获取的2021-2025年加榜区域天气，计算指标与风险关注日期，再接收A的图像结果和B的真实田块资料，产生有理由的巡检需求。

**先读 `加榜梯田_C同学分析报告.pdf`，再看 `分析说明.md`。**前者适合快速理解，后者有更完整的计算与来源说明。

## 已经做好

- 1826天数据核验，保留真实零值，缺测不补零；计算3/7/14/30日降水、湿度与土壤水分变化等。
- 使用2021-2023年建立分季节参考阈值，固定应用到2024-2025年，输出集中降雨、少雨且土壤偏干、持续湿润、偏热、偏冷五类关注信号。
- 生成逐日CSV/JSON、连续候选过程、年度统计、图表及中文报告。
- 适配同学A的`/analyze`、`/analysis/full`、`/anomaly`、`/anomaly/detect`成功返回格式。
- 提供给B的巡检需求JSON、田块/图像元数据约定、本地FastAPI接口和PostgreSQL/PostGIS建表草案。

**尚未实地验证的部分：**没有真实加榜拍摄时间、核准田块边界、灌排记录和异常/正常标签，所以真实田块任务文件目前为空；没有运行同学A的模型、同学B的路径程序或数据库。`examples`中的synthetic田块和图像完全是虚构的接口测试数据。它们不进入年度气象统计，也不能当作论文实测。

## 安装与运行

需要Python 3.10以上。建议在本目录创建独立环境，避免和A的深度学习环境混用：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_analysis.py
.venv\Scripts\python -m pytest -q tests
```

加`--pdf`可重新生成PDF；Windows自动使用系统中文字体。其他系统重新生成中文图表时需安装可用中文字体并在`agri_risk/report.py`配置字体。已交付的PDF和PNG可直接阅读，不需要安装环境。

本包已带整理好的输入CSV，可以离线重算，**不需要再次下载气象数据**。源数据获取与原始JSON/NetCDF仍在之前交付的加榜数据包中；本包`inputs`保留来源、坐标状态、下载请求和输入SHA256。

启动接口，默认只监听本机：

```powershell
.venv\Scripts\python -m uvicorn agri_risk.api:app --host 127.0.0.1 --port 8001
```

浏览器打开 `http://127.0.0.1:8001/docs`。A图像服务默认端口8000，C使用8001。无需先启动A就能查询已生成的历史天气和风险结果。

## 与两位同学对接

按你确认的分工：**A图像 → C气象与证据解释 → B地形与路线**。A仓库文档将图像命名为B模块、平台命名为C模块，本包没有照搬这些字母。

1. A用现有接口分析图像，把成功返回JSON保存下来；另提供`captured_at`、`plot_id`等拍摄元数据。具体字段见`docs/接口与分工.md`。
2. B提供已经核准的真实田块GeoJSON及有时间、来源的灌排问题记录。30米DEM坡度不能直接当作梯田台面坡度。
3. 将三部分组合成`/integration/fuse`请求。可以先用`examples/fusion_synthetic_request.json`做字段演示。
4. C给B的是待执行规划的巡检需求，不含航点高度、药量或飞行许可。B处理航线、障碍、起降点和电量。

也可以不启动服务，在本地处理请求文件：

```powershell
.venv\Scripts\python integrate.py --request examples/fusion_synthetic_request.json --output examples/my_demo_result.json
```

真实请求应删除或设`demo_mode=false`，不得将虚构田块改名后伪装成真实资料。图像时间超过3天、早于/晚于适用日期不符合规则、定位未核准等问题会进入隔离清单；不把错误当成正常结果。

## 重要文件

| 文件 | 用途 |
|---|---|
| `results/weather_clean.csv` | 一天一行的核验后天气 |
| `results/weather_features.csv` | 可直接做图的滚动指标 |
| `results/seasonal_thresholds_2021_2023.csv` | 各月份阈值及参考样本数 |
| `results/risk_daily.csv` / `.json` | 每天每类一条，含实际指标、阈值、状态和依据 |
| `results/candidate_events.csv` / `.json` | 连续触发日期段；起止缺测时保留截断标记 |
| `results/annual_screening_counts.csv` | 各年候选天数和连续过程数 |
| `results/summary.json` | 前端可以直接读取的概要 |
| `results/regional_case_without_field_evidence.json` | 真实天气案例，保持区域级输出 |
| `results/field_inspection_requests_real.json` | 真实田块任务，当前为`[]` |
| `results/A_reference_compatibility_audit.json` | 对A仓库旧示例的检查；这些图像未参与加榜风险融合 |
| `results/figures` | 可放进汇报的静态图 |
| `docs/接口与分工.md` | A/C/B字段和联调顺序 |
| `docs/postgres_schema.sql` | 数据库建表草案，未部署执行 |

## 如何解释风险等级

`screen_level=0`：本规则未触发，**不等于安全**。`1`：一级关注；`2`：更偏离参考期的二级关注；空值：数据不足。它们不是官方预警等级或发病概率。温度阈值是季节相对异常，不是作物生育期的损伤阈值。

P1/P2是巡检排序，需天气信号和真实本地线索配合。图像低绿度可能与光照、物候、营养、缺水或病害有关；本模块保留原因未知，不开具病害诊断或变量施药处方。

历史日数据是事后下载的产品，尚未核对实际发布延迟。因此这份结果是**历史回放筛查**，不是在线预报效果评估。没有真实标签，不能计算可信的灾害预测准确率。下一阶段需要固定规则后前瞻性收集现场证据，再评估。

## 修改与复现

规则定义在`agri_risk/rules.py`；数据清洗在`weather.py`；对接在`integration.py`。`config.json`可调整有雨日阈值、湿润天数和图像证据窗口。改规则需要更新`rule_version`并重新计算；不要反复挑选2025年的规则再把2025年叫独立测试。

`results/runtime_versions.json`记录本次实际运行版本。测试结果及实际覆盖范围见`results/test_run.json`，本次通过29项测试。依赖库有一条TestClient弃用提示，不影响本次结果；并不表示已经完成跨机器联调。
