# 图像分析模块 API 文档

> 状态：阶段 7 已实现，全部端点已用 curl 实测通过。Swagger 见 `/docs`（FastAPI 自动生成）。

## 启动

在 `image_analysis` 根目录：

```bash
cd D:/大创/image_analysis
HF_ENDPOINT=https://hf-mirror.com \
  D:/Anaconda3/envs/DL/python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Swagger：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

## 统一响应格式

成功（HTTP 200）：

```json
{
  "status": "success",
  "message": "success",
  "data": { ... },
  "timestamp": "2026-11-05T12:00:00"
}
```

失败（HTTP 4xx，`detail` 内）：

```json
{
  "status": "error",
  "message": "无法解码影像（仅支持 PNG/JPG/TIFF 等常见格式）",
  "data": null,
  "timestamp": "2026-11-05T12:00:00"
}
```

结果图（掩膜/叠加图等）输出到 `outputs/api/<tag>/`，通过静态目录挂载在 `/files/` 下返回相对路径。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/health` | 健康检查（模型是否加载/设备/类别数） |
| POST | `/upload` | 上传影像，返回 `image_id` + 元信息 |
| POST | `/predict` | 分割预测（类别掩膜 + 类别分布） |
| POST | `/segmentation` | 分割可视化（彩色掩膜 + 叠加图 + 分布） |
| POST | `/vegetation` | 植被指数分析（VARI/ExG/GLI/NDI） |
| POST | `/anomaly` | 异常检测（低绿度 + 裸地） |
| POST | `/analyze` | 综合分析（元信息 + 植被 + 异常 + 分割） |
| GET  | `/history` | 历史分析记录（`?limit=N`） |

### GET /health

```json
{ "model_loaded": false, "device": "cpu", "num_classes": 7 }
```

### POST /upload

`multipart/form-data`，字段 `file`。

```json
{ "image_id": "a1b2c3d4e5f6", "filename": "sample.png",
  "width": 1024, "height": 1024, "channels": 3 }
```

### POST /predict

```json
{
  "pred_mask": "/files/api/<tag>/pred_mask.png",
  "class_distribution": { "background": 0.42, "building": 0.01, "road": 0.02,
    "water": 0.03, "barren": 0.10, "forest": 0.28, "agriculture": 0.14 },
  "num_classes": 7
}
```

### POST /segmentation

```json
{
  "pred_mask": "/files/api/<tag>/color_mask.png",
  "overlay": "/files/api/<tag>/overlay.png",
  "class_distribution": { "...": "..." },
  "num_classes": 7
}
```

### POST /vegetation

```json
{
  "vegetation_fraction_gli": 0.53,
  "vegetation_fraction_vari": 0.49,
  "index_statistics": { "VARI": {...}, "ExG": {...}, "GLI": {...}, "NDI": {...} },
  "note": "RGB 植被指数（VARI/ExG/GLI/NDI），非真实 NDVI（无 NIR 波段）"
}
```

### POST /anomaly

```json
{
  "low_vigor_fraction": 0.03,
  "bare_land_fraction": 0.11,
  "low_vigor_mask": "/files/api/<tag>/low_vigor.png",
  "bare_land_mask": "/files/api/<tag>/bare_land.png"
}
```

### POST /analyze

```json
{
  "filename": "sample.png", "width": 1024, "height": 1024,
  "vegetation": { "...": "..." },
  "anomaly": { "...": "..." },
  "segmentation": { "...": "..." }
}
```

### GET /history

`?limit=50`（默认 50）。

```json
{ "count": 12, "items": [ { "id": "a1b2c3d4e5f6", "type": "analyze",
  "filename": "sample.png", "timestamp": "2026-11-05T12:00:00", "...": "..." } ] }
```

## 示例

```bash
curl -F "file=@sample.png" http://localhost:8000/analyze
curl -F "file=@sample.png" http://localhost:8000/vegetation
curl "http://localhost:8000/history?limit=10"
```

## 实现说明

- 模型懒加载：首次调用预测类端点时加载 SegFormer 最佳权重（`checkpoints_segformer/segformer_best.pth`），线程锁保护。
- 历史记录持久化到 `outputs/api_history.json`（JSON 文件，非 MySQL）。
- 上传影像保存到 `outputs/api/uploads/`；分析产物按 `<tag>` 分目录。
- 中文路径安全：所有图片读写走 `np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `.tofile`。
