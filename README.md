# 数据集存储（成员A：无人机仿真与自适应路径规划）

本目录是**成员A**用到的公开数据与下载脚本，已放在 DeepSeek 工作区根目录下，
与代码目录 `project_A/` 平级。数据下载日期：**2026-11-10**。

> 重要区分：本目录只存放**真实公开数据**（DEM、OSM）。
> 地块边界、建筑、树木、电线、病害风险值等是 `project_A` 中按固定随机种子
> **人工生成的仿真数据**，不在此目录，也不能当作实测结果。

---

## 1. 目录结构

```
数据集存储/
├─ Copernicus_DEM_GLO30/
│  └─ raw/
│     └─ Copernicus_DSM_COG_10_N23_00_E102_00_DEM.tif   43.5 MB  原始 1°×1° DEM（EPSG:4326，约 30 m）
├─ OpenStreetMap/
│  └─ raw/
│     ├─ osm_yuanyang_core.json      888 KB  Overpass API 原始返回（含 tags/geometry）
│     └─ osm_yuanyang_core.geojson   410 KB  转换后矢量，QGIS 可直接打开
├─ QGIS/
│  └─ 安装与使用说明.md                        QGIS（软件，非数据）安装与场景出图步骤
├─ GEE/
│  └─ export_dem_gee.js                        Google Earth Engine 备选导出脚本（需账号授权）
├─ scripts/
│  ├─ download_all.py                          一键重新下载 DEM + OSM
│  └─ convert_osm.py                           Overpass JSON → GeoJSON
├─ manifest.csv                                文件清单、大小、SHA-256
└─ README.md
```

## 2. 研究区

- 位置：云南省红河州**元阳县哈尼梯田核心区**（箐口、多依树、坝达、老虎嘴一带）。
- 经纬度范围：**102.690°E–102.782°E，23.030°N–23.132°N**。
- 覆盖 `project_A` 的 3 套场景：

| 场景 | 中心点 | 范围 | 地形特点 |
|---|---|---|---|
| scene_simple | 102.7481°E, 23.1169°N | 700×700 m | 缓坡梯田，高差约 160 m |
| scene_medium | 102.7625°E, 23.1167°N | 900×900 m | 密集梯田，高差约 370 m |
| scene_complex | 102.7418°E, 23.0821°N | 1100×1100 m | 陡坡梯田，高差约 630 m |

## 3. 数据来源与许可

| 数据 | 来源 | 许可 | 说明 |
|---|---|---|---|
| Copernicus DEM GLO-30 | ESA / Copernicus Programme，AWS Open Data 镜像 `copernicus-dem-30m` | Copernicus DEM 免费开放，需注明来源 | 1 角秒（约 30 m）DSM；本目录为 N23/E102 瓦片 |
| OpenStreetMap | © OpenStreetMap contributors，Overpass API 抽取 | **ODbL 1.0** | 道路、水系、建筑、林地、农田等；使用时保留署名 |
| QGIS | QGIS.org | GPL | 桌面软件，未在本目录存放安装包 |
| GEE DEM 目录 | Google Earth Engine `COPERNICUS/DEM/GLO30` | 遵循 GEE 条款 | 需账号授权，仅提供导出脚本 |

原始 DEM 下载地址（无需账号）：

```
https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N23_00_E102_00_DEM/Copernicus_DSM_COG_10_N23_00_E102_00_DEM.tif
```

## 4. 为什么 OSM 用 Overpass 而不是 Geofabrik 全量

Geofabrik 的中国全量 PBF 约 1.4 GB，而本研究只需元阳约 10 km×11 km 的一小块。
Overpass API 可以按 bbox 精确抽取同名 OSM 数据，许可仍是 ODbL，体积只有 0.9 MB。
如需全国/全省底图，再到 <https://download.geofabrik.de/asia/china.html> 下载。

## 5. 数据如何使用

1. `project_A/scripts/01_build_scenes.py` 读取本目录的 DEM 瓦片与 OSM JSON，
   重投影到 **EPSG:32648（UTM 48N）**、重采样到 **5 m** 规划网格。
2. 在真实地形上人工布设地块、建筑、树木、电线（固定随机种子，可复现），
   输出到 `project_A/data/scenes/scene_*/`。
3. QGIS 查看方法见 `QGIS/安装与使用说明.md`。

> 30 m DEM 无法表达田埂、沟渠、小梯田的细节。项目按“**真实地形底图 + 局部 5 m 仿真网格 + 明确标注人工障碍物**”的方式处理，
> 避免把仿真地块误称为真实测绘结果。

## 6. 重新下载 / 校验

```bash
# 在本目录下
python scripts/download_all.py            # DEM + OSM（已存在则跳过）
python scripts/download_all.py --dem-only
python scripts/download_all.py --osm-only

# 校验文件完整性
python -c "import csv,hashlib; [print(r['path'], hashlib.sha256(open(r['path'],'rb').read()).hexdigest()==r['sha256']) for r in csv.DictReader(open('manifest.csv',encoding='utf-8-sig'))]"
```

> Overpass 是公共公益服务，偶尔 502/超时；脚本内置 3 个镜像与重试。
> 若仍失败，可在 <https://overpass-turbo.eu/> 粘贴 `scripts/download_all.py` 中的查询语句导出。

## 7. 引用建议

- Copernicus DEM：© ESA / Copernicus Programme，Copernicus DEM GLO-30，2026 年下载。
- 地图数据：© OpenStreetMap contributors，ODbL 1.0。
- 论文方法：Gkelios et al., *Beyond Coverage Path Planning: Can UAV Swarms Perfect Scattered Regions Inspections?*, arXiv:2512.23257 / RAS 2026。
