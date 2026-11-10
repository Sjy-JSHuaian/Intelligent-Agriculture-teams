"""FastAPI 集成测试（阶段 8）。

覆盖：健康检查、模型信息、上传（正常/空文件/非法格式）、分割、植被、异常、
综合分析、统一 JSON 格式、中文路径、CPU 推理。

说明：
- 分割类端点（/predict /segmentation /segment /analyze /analysis/full）依赖
  SegFormer 权重，测试中用轻量 mock 模型替代（monkeypatch api.main.get_model），
  属于「mock 测试」而非真实推理测试。
- 植被/异常/上传/健康检查/模型信息为「真实计算测试」，不依赖权重与 GPU。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch

import api.main as api
from datasets.preprocessing import read_image


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _png_bytes(h: int = 32, w: int = 32) -> bytes:
    """生成一张偏绿的 RGB PNG 字节（内存中，无需真实文件）。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 1] = 150          # 绿色分量，便于植被/异常分析
    img[..., 0] = 60
    img[..., 2] = 40
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class _DummySegModel:
    """mock 分割模型：所有像素预测为 forest（训练类 5）。"""

    def __call__(self, x):
        B, _, H, W = x.shape
        logits = torch.zeros(B, api.NUM_CLASSES, H, W)
        logits[:, 5, :, :] = 100.0
        return logits

    def eval(self):
        return self

    def parameters(self):
        return []


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient，将输出目录与历史文件重定向到临时目录，避免污染真实 outputs。"""
    from fastapi.testclient import TestClient
    monkeypatch.setattr(api, "OUTPUTS", tmp_path)
    monkeypatch.setattr(api, "HISTORY_PATH", tmp_path / "api_history.json")
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def mock_model(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: _DummySegModel())


# --------------------------------------------------------------------------- #
# 健康检查 / 模型信息
# --------------------------------------------------------------------------- #
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["status"] == "success"
    assert body["data"]["device"] == "cpu"
    assert body["data"]["num_classes"] == 7


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["model"] == "SegFormer-B0"
    assert d["num_classes"] == 7
    assert d["device"] == "cpu"
    assert "checkpoint_exists" in d and "loaded" in d


# --------------------------------------------------------------------------- #
# 上传
# --------------------------------------------------------------------------- #
def test_upload_valid(client):
    r = client.post("/upload", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["width"] == 32 and d["height"] == 32 and d["channels"] == 3
    assert d["image_id"]


def test_upload_empty_file(client):
    r = client.post("/upload", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"
    assert "空" in r.json()["detail"]["message"]


def test_upload_invalid_image(client):
    r = client.post("/upload", files={"file": ("bad.png", b"not-an-image", "image/png")})
    assert r.status_code == 400
    assert r.json()["detail"]["status"] == "error"


# --------------------------------------------------------------------------- #
# 分割（mock 模型）
# --------------------------------------------------------------------------- #
def test_predict_mock(client, mock_model):
    r = client.post("/predict", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["model"] == "SegFormer-B0"
    assert d["num_classes"] == 7
    assert d["class_distribution"]["forest"] == 1.0   # mock 全部预测为 forest
    assert d["pred_mask"].startswith("/files/")


def test_segment_alias(client, mock_model):
    r = client.post("/segment", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert "pred_mask" in r.json()["data"]


def test_segmentation_mock(client, mock_model):
    r = client.post("/segmentation", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()["data"]
    assert "overlay" in d and "pred_mask" in d


# --------------------------------------------------------------------------- #
# 植被 / 异常（真实计算，无需模型）
# --------------------------------------------------------------------------- #
def test_vegetation_real(client):
    r = client.post("/vegetation", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()["data"]
    assert 0.0 <= d["vegetation_fraction_gli"] <= 1.0
    assert set(d["index_statistics"]) == {"VARI", "ExG", "GLI", "NDI"}
    assert "非真实 NDVI" in d["note"]


def test_vegetation_alias(client):
    r = client.post("/vegetation/analyze",
                    files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert "index_statistics" in r.json()["data"]


def test_anomaly_real(client):
    r = client.post("/anomaly", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()["data"]
    assert 0.0 <= d["low_vigor_fraction"] <= 1.0
    assert 0.0 <= d["bare_land_fraction"] <= 1.0


def test_anomaly_alias(client):
    r = client.post("/anomaly/detect",
                    files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert "bare_land_fraction" in r.json()["data"]


# --------------------------------------------------------------------------- #
# 综合分析（mock 分割 + 真实植被/异常）
# --------------------------------------------------------------------------- #
def test_analyze_full_mock(client, mock_model):
    r = client.post("/analyze", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()["data"]
    assert set(d) == {"filename", "width", "height", "vegetation", "anomaly", "segmentation"}
    assert d["width"] == 32 and d["height"] == 32


def test_analysis_full_alias(client, mock_model):
    r = client.post("/analysis/full",
                    files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert "segmentation" in r.json()["data"]


# --------------------------------------------------------------------------- #
# 统一 JSON 格式 / 历史
# --------------------------------------------------------------------------- #
def test_unified_json_schema(client):
    """所有响应均包含 success/status/message/data/timestamp 顶层字段。"""
    r = client.get("/health")
    body = r.json()
    for key in ("success", "status", "message", "data", "timestamp"):
        assert key in body


def test_history(client):
    client.post("/upload", files={"file": ("a.png", _png_bytes(), "image/png")})
    client.post("/vegetation", files={"file": ("b.png", _png_bytes(), "image/png")})
    r = client.get("/history")
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["count"] >= 2
    assert all("id" in it and "timestamp" in it for it in d["items"])


# --------------------------------------------------------------------------- #
# 中文路径 / 序列化
# --------------------------------------------------------------------------- #
def test_chinese_path_roundtrip(tmp_path):
    """中文路径下 PNG 写（imencode+tofile）与读（np.fromfile+imdecode）往返。"""
    d = tmp_path / "中文目录"
    d.mkdir()
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 1] = 200
    p = d / "测试影像.png"
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(p))                       # 中文路径写
    got = read_image(str(p))                 # 中文路径读
    assert got.shape == (8, 8, 3)
    assert got[0, 0, 1] == 200


def test_no_numpy_types_in_json(client):
    """返回 JSON 不应出现 NumPy 类型（TestClient 的 json() 反序列化即验证）。"""
    r = client.post("/vegetation", files={"file": ("t.png", _png_bytes(), "image/png")})
    body = r.json()                          # 若含 NumPy 类型，此处会抛 TypeError
    stats = body["data"]["index_statistics"]["GLI"]
    assert isinstance(stats["mean"], float)
