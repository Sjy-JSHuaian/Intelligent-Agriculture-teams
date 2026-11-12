# B 模块 API 集成文档（供 C 模块 / Web 平台调用）

> 版本：0.2.0（阶段 8）· 全部端点已用 pytest TestClient 与 curl 实测通过。

---

## 1. 服务地址

启动后（见 `DEPLOYMENT_GUIDE.md`）：

- 根地址：`http://<host>:8000`
- Swagger 交互文档：`http://<host>:8000/docs`
- OpenAPI JSON：`http://<host>:8000/openapi.json`

---

## 2. 统一返回格式

**成功（HTTP 200）**：

```json
{
  "success": true,
  "status": "success",
  "message": "success",
  "data": { "...": "..." },
  "timestamp": "2026-11-06T12:00:00"
}
```

**失败（HTTP 4xx/5xx）**，错误体在 `detail` 字段内：

```json
{
  "detail": {
    "success": false,
    "status": "error",
    "message": "上传文件为空",
    "data": null,
    "timestamp": "2026-11-06T12:00:00"
  }
}
```

约定：**所有数值均来自真实计算**；未实现的字段不填充，不返回占位假值。

---

## 3. 图片上传方式

- 方式：`multipart/form-data`，表单字段名固定为 **`file`**。
- 支持格式：`PNG / JPG / JPEG`（RGB 常见格式）。
- 结果图访问：返回的 `mask_path` / `overlay` / `*_mask` 等均为相对路径，
  形如 `/files/api/<tag>/xxx.png`；通过静态挂载直接 `GET` 该路径即可下载，
  即 `http://<host>:8000/files/api/<tag>/xxx.png`。

---

## 4. 端点一览

| 方法 | 路径 | 说明 | 依赖模型 |
|------|------|------|---------|
| GET | `/health` | 健康检查 | 否 |
| GET | `/model/info` | 模型信息查询 | 否 |
| POST | `/upload` | 上传影像，返回 image_id + 元信息 | 否 |
| POST | `/predict` | 语义分割预测（掩膜 + 类别分布） | 是 |
| POST | `/segment` | `/predict` 的 RESTful 别名 | 是 |
| POST | `/segmentation` | 分割可视化（叠加图 + 彩色掩膜 + 分布） | 是 |
| POST | `/vegetation` | 植被指数分析（RGB 指数） | 否 |
| POST | `/vegetation/analyze` | `/vegetation` 别名 | 否 |
| POST | `/anomaly` | 异常检测（低绿度 + 裸地） | 否 |
| POST | `/anomaly/detect` | `/anomaly` 别名 | 否 |
| POST | `/analyze` | 综合分析（元信息 + 植被 + 异常 + 分割） | 是 |
| POST | `/analysis/full` | `/analyze` 别名 | 是 |
| GET | `/history` | 历史分析记录（`?limit=N`） | 否 |

> 「依赖模型」为「是」的端点首次调用会懒加载 SegFormer 权重；权重缺失时返回 **503**。

---

## 5. 各端点返回示例

### GET /health

```json
{ "success": true, "status": "success", "message": "success",
  "data": { "model_loaded": true, "device": "cpu", "num_classes": 7, "model": "SegFormer-B0" },
  "timestamp": "..." }
```

### GET /model/info

```json
{ "success": true, "data": {
    "model": "SegFormer-B0", "num_classes": 7, "device": "cpu",
    "checkpoint": ".../checkpoints_segformer/segformer_best.pth",
    "checkpoint_exists": true, "loaded": true, "params": 3720000, "params_mb": 14.9 } }
```

### POST /upload

```bash
curl -F "file=@sample.png" http://localhost:8000/upload
```

```json
{ "success": true, "data": { "image_id": "a1b2c3d4e5f6", "filename": "sample.png",
  "width": 1024, "height": 1024, "channels": 3 } }
```

### POST /predict（及 /segment）

```json
{ "success": true, "data": {
    "model": "SegFormer-B0",
    "pred_mask": "/files/api/<tag>/pred_mask.png",
    "class_distribution": { "background": 0.29, "building": 0.20, "road": 0.07,
      "water": 0.05, "barren": 0.0, "forest": 0.0, "agriculture": 0.38 },
    "num_classes": 7 } }
```

### POST /segmentation

```json
{ "success": true, "data": {
    "model": "SegFormer-B0",
    "pred_mask": "/files/api/<tag>/color_mask.png",
    "overlay": "/files/api/<tag>/overlay.png",
    "class_distribution": { "...": "..." }, "num_classes": 7 } }
```

### POST /vegetation（及 /vegetation/analyze）

```json
{ "success": true, "data": {
    "vegetation_fraction_gli": 0.53,
    "vegetation_fraction_vari": 0.49,
    "index_statistics": { "VARI": {"mean":0.1,"std":0.08,"min":-0.8,"max":0.9},
      "ExG": {...}, "GLI": {...}, "NDI": {...} },
    "note": "RGB 植被指数（VARI/ExG/GLI/NDI），非真实 NDVI（无 NIR 波段）" } }
```

### POST /anomaly（及 /anomaly/detect）

```json
{ "success": true, "data": {
    "low_vigor_fraction": 0.03, "bare_land_fraction": 0.11,
    "low_vigor_mask": "/files/api/<tag>/low_vigor.png",
    "bare_land_mask": "/files/api/<tag>/bare_land.png" } }
```

### POST /analyze（及 /analysis/full）

```json
{ "success": true, "data": {
    "filename": "sample.png", "width": 1024, "height": 1024,
    "vegetation": { "...": "见 /vegetation" },
    "anomaly": { "...": "见 /anomaly" },
    "segmentation": { "...": "见 /segmentation" } } }
```

### GET /history

```bash
curl "http://localhost:8000/history?limit=10"
```

```json
{ "success": true, "data": { "count": 12,
  "items": [ { "id": "a1b2c3d4e5f6", "type": "analyze", "filename": "sample.png",
               "timestamp": "...", "...": "..." } ] } }
```

---

## 6. 错误码

| HTTP | 场景 | message 示例 |
|------|------|-------------|
| 400 | 空文件 | `上传文件为空` |
| 400 | 非法/无法解码格式 | `无法解码影像（仅支持 PNG/JPG/JPEG 等常见 RGB 格式）` |
| 503 | 模型权重缺失/加载失败 | `模型权重缺失: .../segformer_best.pth（需先训练或放入权重文件）` |

---

## 7. C 模块调用方式（建议）

1. **上传+分析**：C 模块前端拿到用户上传的影像后，`POST /analysis/full`（`multipart`，字段 `file`），
   一次拿到植被 + 异常 + 分割三类结果。
2. **结果图**：将返回的 `/files/...` 相对路径拼到服务地址前，直接在 `<img>` 或地图叠加层使用。
3. **历史**：`GET /history?limit=N` 拉取分析记录列表。
4. **模型状态**：启动前 `GET /health` + `GET /model/info` 确认模型可用。

示例（Python requests）：

```python
import requests
r = requests.post("http://localhost:8000/analysis/full",
                  files={"file": ("sample.png", open("sample.png", "rb"), "image/png")})
data = r.json()["data"]
overlay_url = "http://localhost:8000" + data["segmentation"]["overlay"]
veg_frac = data["vegetation"]["vegetation_fraction_gli"]
```

---

## 8. 地理坐标预留（与 A/C 模块衔接）

- 当前模型处理 RGB 影像，**不产生**经纬度；普通 JPG/PNG 无地理参考，返回中**不含**伪造坐标字段。
- 若 A 模块提供带地理参考的 GeoTIFF，B 模块在数据读取阶段保留 CRS 与仿射变换
  （见 `DATA_FORMAT_SPEC.md`），为后续 C 模块地图可视化预留地理坐标字段。
