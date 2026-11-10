# 数据格式规范（Data Format Spec）

> 明确 B 模块输入/输出的图像格式、通道、尺寸、类别编码与 JSON 字段，
> 供 A（数据采集）、C（平台）模块对齐。所有约定以本模块实际实现为准。

---

## 1. 输入图像

| 项 | 规范 |
|----|------|
| 格式 | PNG / JPG / JPEG（RGB 常见格式；GeoTIFF 走 rasterio 读取） |
| 通道 | **3 通道 RGB**（无 NIR 波段，本模块不计算真实 NDVI） |
| 内存表示 | `HWC`、`uint8`、值域 [0, 255] |
| 颜色顺序 | RGB（读入时已由 OpenCV BGR 转为 RGB） |
| 原始尺寸 | LoveDA 为 1024×1024；模型输入统一 Resize 到 256×256 |
| 中文路径 | 读写用 `np.fromfile+imdecode` / `imencode+tofile` |

---

## 2. 类别编码（LoveDA，最重要）

**原始掩膜取值 0-7**（官方定义）：

| 原始值 | 名称 | 说明 |
|:---:|------|------|
| 0 | no-data | **忽略**（不参与训练/指标） |
| 1 | background | 背景 |
| 2 | building | 建筑 |
| 3 | road | 道路 |
| 4 | water | 水体 |
| 5 | barren | 裸地 |
| 6 | forest | 森林 |
| 7 | agriculture | **农田**（梯田核心目标类） |

**训练重映射**（`datasets/dataset_loader.py::remap_mask`）：

- 原始 1-7 → 训练 0-6（`NUM_CLASSES=7`）
- 原始 0(no-data) → 255（`IGNORE_INDEX=255`，损失与指标均忽略）

| 训练值 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 255 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 名称 | background | building | road | water | barren | forest | agriculture | ignore |

> API 返回的类别分布键名（`class_distribution`）使用**训练值对应名称**
> （background/building/road/water/barren/forest/agriculture）。

---

## 3. 输出掩膜

- 预测掩膜：`H×W`、`uint8`，取值 0-6（训练编码）。
- 可视化掩膜：转回原始 1-7 配色（`visualization/segmentation_visualizer.py::LOVEDA_COLORS`）。
- 叠加图：原图与彩色掩膜按 alpha=0.5 融合。

---

## 4. 植被指数（RGB，非 NDVI）

| 指数 | 公式 | 值域 | 说明 |
|------|------|------|------|
| VARI | (G−R)/(G+R−B) | [−1,1]（已 clip） | 可见光抗大气植被指数 |
| ExG | 2G−R−B | [−2,2] | 过绿指数 |
| GLI | (2G−R−B)/(2G+R+B) | [−1,1] | 绿叶指数（植被判定主用） |
| NDI | (G−R)/(G+R) | [−1,1] | 归一化差异指数 |

> 阈值均为经验值（`analysis/vegetation.py`），待加榜梯田真实影像到位后校准。
> **明确标注：RGB 指数，不是真实 NDVI。**

---

## 5. 坐标信息（地理参考）

- **GeoTIFF 输入**：`datasets/preprocessing.py` 通过 rasterio 读取时保留
  CRS（坐标参考系）与仿射变换（affine transform）；切片时记录切片左上角
  原图坐标，用于后续像素↔地理坐标映射。
- **普通 JPG/PNG 输入**：无地理参考信息，**不伪造经纬度**；API 返回中不含坐标字段。
- **为 C 模块地图可视化预留**：后续若 A 模块提供带地理参考影像，B 模块在输出
  JSON 中可增加 `crs`、`transform`、`pixel_to_geo` 字段（当前未启用，因数据未到位）。

---

## 6. JSON 字段规范

所有 API 返回值遵循统一外层结构（见 `API_INTEGRATION.md`），`data` 内字段：

| 端点 | 字段 | 类型 | 说明 |
|------|------|------|------|
| /upload | image_id / filename / width / height / channels | str/str/int/int/int | 元信息 |
| /predict | model / pred_mask / class_distribution / num_classes | str/str/object/int | 分割预测 |
| /segmentation | model / pred_mask / overlay / class_distribution / num_classes | — | 分割可视化 |
| /vegetation | vegetation_fraction_gli / _vari / index_statistics / note | float/float/object/str | 植被 |
| /anomaly | low_vigor_fraction / bare_land_fraction / low_vigor_mask / bare_land_mask | float/float/str/str | 异常 |
| /analyze | filename / width / height / vegetation / anomaly / segmentation | — | 综合 |

> 数值字段一律 Python 原生 `float`/`int`（已 `float()` 显式转换），
> **不返回 NumPy 类型**（无法 JSON 序列化）。numpy 数组（指数图/掩膜）仅落盘为
> PNG，不进入 JSON。

---

## 7. 数据一致性检查（多源数据接入时）

当 A 模块提供加榜梯田影像时，需核对：

- [ ] CRS 是否一致（不一致需重投影）
- [ ] 空间分辨率是否一致（不一致需重采样）
- [ ] 时间是否一致（多时相变化检测需说明）
- [ ] 波段数量与含义（是否含 NIR）
- [ ] 是否按地理区域/场景划分 train/val/test，避免同源切片泄漏
