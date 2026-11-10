"""加榜梯田智慧农业 —— 图像智能分析 REST API。

统一返回格式（成功，HTTP 200）：
    {"success": true, "status": "success", "message": str, "data": {...}, "timestamp": iso}

失败（HTTP 4xx/5xx，detail 内）：
    {"success": false, "status": "error", "message": str, "data": null, "timestamp": iso}

启动（在 image_analysis 根目录）：
    uvicorn api.main:app --host 0.0.0.0 --port 8000

端点：
    GET  /health            健康检查
    GET  /model/info        模型信息查询（名称/类别数/权重路径/加载状态）
    POST /upload            上传影像（返回元信息 + image_id）
    POST /analyze           上传并综合分析（植被+异常+分割）
    POST /predict           语义分割预测（类别掩膜 + 类别分布）
    POST /segmentation      分割可视化（叠加图 + 颜色分布 + 统计）
    POST /vegetation        植被指数分析（VARI/ExG/GLI/NDI，非 NDVI）
    POST /anomaly           异常检测（低绿度 + 裸地）
    GET  /history           历史分析记录

兼容别名（与 C 模块约定的 RESTful 风格，语义同上）：
    POST /segment            == /predict
    POST /vegetation/analyze == /vegetation
    POST /anomaly/detect     == /anomaly
    POST /analysis/full      == /analyze
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from analysis.anomaly import detect_bare_land, detect_low_vigor
from analysis.vegetation import analyze_vegetation, vegetation_fraction
from datasets.augmentation import get_val_augmentation
from datasets.dataset_loader import NUM_CLASSES, TRAIN_CLASSES
from visualization.segmentation_visualizer import mask_to_color, overlay

app = FastAPI(title="加榜梯田图像智能分析 API", version="0.2.0")

OUTPUTS = ROOT / "outputs"
HISTORY_PATH = OUTPUTS / "api_history.json"
UPLOAD_DIR = OUTPUTS / "api" / "uploads"
MODEL_CONFIG = ROOT / "configs" / "model_segformer.yaml"
MODEL_CHECKPOINT = ROOT / "checkpoints_segformer" / "segformer_best.pth"
MODEL_NAME = "SegFormer-B0"

for d in (UPLOAD_DIR,):
    d.mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=str(OUTPUTS)), name="files")

# --------------------------------------------------------------------------- #
# 统一响应
# --------------------------------------------------------------------------- #
def ok(data, message: str = "success") -> dict:
    return {"success": True, "status": "success", "message": message,
            "data": data, "timestamp": datetime.now().isoformat()}


def fail(message: str, code: int = 400) -> HTTPException:
    return HTTPException(status_code=code, detail={
        "success": False, "status": "error", "message": message,
        "data": None, "timestamp": datetime.now().isoformat()})


class ModelUnavailable(RuntimeError):
    """模型权重缺失或加载失败（区分于输入错误，返回 503）。"""


# --------------------------------------------------------------------------- #
# 模型懒加载（SegFormer，最佳模型）
# --------------------------------------------------------------------------- #
_model = None
_model_lock = threading.Lock()


def get_model():
    """懒加载 SegFormer 最佳模型；加载失败抛 ModelUnavailable。"""
    global _model
    with _model_lock:
        if _model is None:
            try:
                from models import build_model
                if not MODEL_CONFIG.exists():
                    raise ModelUnavailable(f"模型配置缺失: {MODEL_CONFIG}")
                with open(MODEL_CONFIG, encoding="utf-8") as f:
                    model_cfg = yaml.safe_load(f)["model"]
                model = build_model(model_cfg)
                if not MODEL_CHECKPOINT.exists():
                    raise ModelUnavailable(
                        f"模型权重缺失: {MODEL_CHECKPOINT}（需先训练或放入权重文件）")
                ckpt = torch.load(MODEL_CHECKPOINT, map_location="cpu")
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                _model = model
            except ModelUnavailable:
                raise
            except Exception as e:  # 任何其它加载错误都转成清晰的服务不可用
                raise ModelUnavailable(f"模型加载失败: {type(e).__name__}: {e}") from e
    return _model


def get_model_info() -> dict:
    """模型信息（不触发权重加载，仅报告配置与存在性）。"""
    params = None
    if _model is not None:
        params = sum(p.numel() for p in _model.parameters())
    return {
        "model": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "device": "cpu",
        "checkpoint": str(MODEL_CHECKPOINT),
        "checkpoint_exists": MODEL_CHECKPOINT.exists(),
        "loaded": _model is not None,
        "params": params,
        "params_mb": round(params * 4 / 1e6, 1) if params is not None else None,
    }


def decode_image(data: bytes) -> np.ndarray:
    """上传字节 -> HWC uint8 RGB。空文件或非法格式抛 ValueError。"""
    if not data:
        raise ValueError("上传文件为空")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码影像（仅支持 PNG/JPG/JPEG 等常见 RGB 格式）")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_bgr(img_bgr: np.ndarray, rel_path: str) -> str:
    """保存 BGR 图为 PNG（中文路径安全），返回可访问的相对路径。"""
    abs_path = OUTPUTS / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    ok_, buf = cv2.imencode(".png", img_bgr)
    if not ok_:
        raise IOError(f"编码失败: {abs_path}")
    buf.tofile(str(abs_path))
    return f"/files/{rel_path}"


# --------------------------------------------------------------------------- #
# 分析服务
# --------------------------------------------------------------------------- #
def _predict_mask(image: np.ndarray) -> np.ndarray:
    """分割预测，返回原尺寸 0-6 类别掩膜。"""
    val_aug = get_val_augmentation((256, 256))
    inp = val_aug(image=image)["image"]                       # HWC float 256
    x = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        logits = get_model()(x)
    pred = logits.argmax(1).squeeze(0).numpy()                # 0-6 @ 256
    H, W = image.shape[:2]
    return cv2.resize(pred.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST)        # 0-6 @ 原尺寸


def _distribution(pred_full: np.ndarray) -> dict:
    total = pred_full.size
    return {TRAIN_CLASSES[c]: round(float((pred_full == c).sum()) / total, 4)
            for c in range(NUM_CLASSES)}


def run_predict(image: np.ndarray, tag: str) -> dict:
    """分割预测，返回预测掩膜与类别分布。"""
    pred_full = _predict_mask(image)
    pred_raw = pred_full.astype(np.int64) + 1                 # -> 原始 1-7 配色
    mask_path = _save_bgr(pred_raw.astype(np.uint8), f"api/{tag}/pred_mask.png")
    return {"model": MODEL_NAME, "pred_mask": mask_path,
            "class_distribution": _distribution(pred_full),
            "num_classes": NUM_CLASSES}


def run_segmentation(image: np.ndarray, tag: str) -> dict:
    """分割 + 叠加可视化。"""
    pred_full = _predict_mask(image)
    pred_raw = pred_full.astype(np.int64) + 1
    overlay_path = _save_bgr(cv2.cvtColor(overlay(image, pred_raw), cv2.COLOR_RGB2BGR),
                             f"api/{tag}/overlay.png")
    color_path = _save_bgr(cv2.cvtColor(mask_to_color(pred_raw), cv2.COLOR_RGB2BGR),
                           f"api/{tag}/color_mask.png")
    return {"model": MODEL_NAME, "pred_mask": color_path, "overlay": overlay_path,
            "class_distribution": _distribution(pred_full), "num_classes": NUM_CLASSES}


def run_vegetation(image: np.ndarray, tag: str) -> dict:
    """植被指数分析（RGB 指数，非 NDVI）。"""
    result = analyze_vegetation(image)
    return {
        "vegetation_fraction_gli": vegetation_fraction(image, "GLI"),
        "vegetation_fraction_vari": vegetation_fraction(image, "VARI"),
        "index_statistics": result["index_statistics"],
        "note": "RGB 植被指数（VARI/ExG/GLI/NDI），非真实 NDVI（无 NIR 波段）",
    }


def run_anomaly(image: np.ndarray, tag: str) -> dict:
    """异常检测：低绿度斑块 + 裸地。"""
    low_vigor = detect_low_vigor(image)
    bare = detect_bare_land(image)
    lv_path = _save_bgr((low_vigor.astype(np.uint8) * 255),
                        f"api/{tag}/low_vigor.png")
    bare_path = _save_bgr((bare.astype(np.uint8) * 255),
                          f"api/{tag}/bare_land.png")
    return {
        "low_vigor_fraction": round(float(low_vigor.mean()), 4),
        "bare_land_fraction": round(float(bare.mean()), 4),
        "low_vigor_mask": lv_path,
        "bare_land_mask": bare_path,
    }


# --------------------------------------------------------------------------- #
# 历史记录
# --------------------------------------------------------------------------- #
_history_lock = threading.Lock()


def add_history(entry: dict) -> None:
    with _history_lock:
        history = []
        if HISTORY_PATH.exists():
            try:
                history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = []
        entry["id"] = uuid.uuid4().hex[:12]
        entry["timestamp"] = datetime.now().isoformat()
        history.append(entry)
        HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                                encoding="utf-8")


# --------------------------------------------------------------------------- #
# 端点核心实现（供主路径与别名路径复用）
# --------------------------------------------------------------------------- #
async def _do_upload(file: UploadFile):
    image = decode_image(await file.read())
    image_id = uuid.uuid4().hex[:12]
    _save_bgr(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
              f"api/uploads/{image_id}_{file.filename}")
    add_history({"type": "upload", "filename": file.filename})
    return {"image_id": image_id, "filename": file.filename,
            "width": image.shape[1], "height": image.shape[0],
            "channels": image.shape[2]}


async def _do_predict(file: UploadFile):
    image = decode_image(await file.read())
    tag = uuid.uuid4().hex[:12]
    res = run_predict(image, tag)
    add_history({"type": "predict", "filename": file.filename,
                 "class_distribution": res["class_distribution"]})
    return res


async def _do_segmentation(file: UploadFile):
    image = decode_image(await file.read())
    tag = uuid.uuid4().hex[:12]
    res = run_segmentation(image, tag)
    add_history({"type": "segmentation", "filename": file.filename,
                 "class_distribution": res["class_distribution"]})
    return res


async def _do_vegetation(file: UploadFile):
    image = decode_image(await file.read())
    tag = uuid.uuid4().hex[:12]
    res = run_vegetation(image, tag)
    add_history({"type": "vegetation", "filename": file.filename,
                 "vegetation_fraction": res["vegetation_fraction_gli"]})
    return res


async def _do_anomaly(file: UploadFile):
    image = decode_image(await file.read())
    tag = uuid.uuid4().hex[:12]
    res = run_anomaly(image, tag)
    add_history({"type": "anomaly", "filename": file.filename, **res})
    return res


async def _do_analyze(file: UploadFile):
    image = decode_image(await file.read())
    tag = uuid.uuid4().hex[:12]
    data = {
        "filename": file.filename,
        "width": image.shape[1], "height": image.shape[0],
        "vegetation": run_vegetation(image, tag),
        "anomaly": run_anomaly(image, tag),
        "segmentation": run_segmentation(image, tag),
    }
    add_history({"type": "analyze", "filename": file.filename,
                 "vegetation_fraction": data["vegetation"]["vegetation_fraction_gli"],
                 "low_vigor_fraction": data["anomaly"]["low_vigor_fraction"]})
    return data


# --------------------------------------------------------------------------- #
# 端点
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return ok({"model_loaded": _model is not None, "device": "cpu",
               "num_classes": NUM_CLASSES, "model": MODEL_NAME})


@app.get("/model/info")
def model_info():
    return ok(get_model_info())


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        return ok(await _do_upload(file))
    except ValueError as e:
        raise fail(str(e))


@app.post("/predict")
@app.post("/segment")
async def predict(file: UploadFile = File(...)):
    try:
        return ok(await _do_predict(file))
    except ValueError as e:
        raise fail(str(e))
    except ModelUnavailable as e:
        raise fail(str(e), code=503)


@app.post("/segmentation")
async def segmentation(file: UploadFile = File(...)):
    try:
        return ok(await _do_segmentation(file))
    except ValueError as e:
        raise fail(str(e))
    except ModelUnavailable as e:
        raise fail(str(e), code=503)


@app.post("/vegetation")
@app.post("/vegetation/analyze")
async def vegetation(file: UploadFile = File(...)):
    try:
        return ok(await _do_vegetation(file))
    except ValueError as e:
        raise fail(str(e))


@app.post("/anomaly")
@app.post("/anomaly/detect")
async def anomaly(file: UploadFile = File(...)):
    try:
        return ok(await _do_anomaly(file))
    except ValueError as e:
        raise fail(str(e))


@app.post("/analyze")
@app.post("/analysis/full")
async def analyze(file: UploadFile = File(...)):
    """上传并综合分析：元信息 + 植被 + 异常 + 分割。"""
    try:
        return ok(await _do_analyze(file))
    except ValueError as e:
        raise fail(str(e))
    except ModelUnavailable as e:
        raise fail(str(e), code=503)


@app.get("/history")
def history(limit: int = 50):
    data = []
    if HISTORY_PATH.exists():
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    return ok({"count": len(data), "items": data[-limit:]})
