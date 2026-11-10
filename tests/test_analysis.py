"""阶段 6 植被/异常分析单元测试（使用合成影像，无需真实数据）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.anomaly import detect_bare_land, detect_change, detect_low_vigor
from analysis.vegetation import (analyze_vegetation, vegetation_fraction,
                                 vegetation_mask)


def _green(h=32, w=32):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 1] = 255
    return img


def _red(h=32, w=32):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 255
    return img


def test_vegetation_fraction_green_is_high():
    assert vegetation_fraction(_green()) > 0.95


def test_vegetation_fraction_red_is_low():
    assert vegetation_fraction(_red()) < 0.05


def test_vegetation_mask_shape_and_dtype():
    m = vegetation_mask(_green())
    assert m.shape == (32, 32)
    assert m.dtype == bool


def test_low_vigor_returns_bool_mask():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 40; img[..., 1] = 140; img[..., 2] = 40   # 健康绿
    img[10:20, 10:20, 0] = 120
    img[10:20, 10:20, 1] = 90
    img[10:20, 10:20, 2] = 50                               # 胁迫斑块（偏棕）
    a = detect_low_vigor(img)
    assert a.shape == (32, 32) and a.dtype == bool
    assert a[10:20, 10:20].sum() > 0          # 低绿度区应被标出
    assert a[0:5, 0:5].sum() == 0             # 正常绿色区不应标出


def test_bare_land_returns_bool_mask():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 150; img[..., 1] = 90; img[..., 2] = 60   # 棕土色（R 占优）
    b = detect_bare_land(img)
    assert b.shape == (32, 32) and b.dtype == bool
    assert b.mean() > 0.5


def test_change_identical_images_is_zero():
    img = _green()
    assert detect_change(img, img).sum() == 0


def test_change_different_images_is_nonzero():
    a = _green()
    b = _red()
    assert detect_change(a, b).sum() > 0


def test_change_mismatched_shape_raises():
    with pytest.raises(ValueError):
        detect_change(_green(32, 32), _red(16, 16))


def test_analyze_vegetation_keys():
    r = analyze_vegetation(_green())
    assert set(r) == {"indices", "index_statistics", "vegetation_fraction", "vegetation_mask"}
    assert set(r["indices"]) == {"VARI", "ExG", "GLI", "NDI"}


def test_vari_is_bounded():
    from datasets.preprocessing import compute_vari
    # 含偏蓝像素（G+R≈B，分母趋近 0）的影像不应产生爆炸值
    rng = np.random.default_rng(0)
    img = rng.random((32, 32, 3)).astype(np.float32)
    img[5, 5] = [0.1, 0.2, 0.3]   # G + R - B ≈ 0
    vari = compute_vari(img)
    assert np.isfinite(vari).all()
    assert vari.min() >= -1.0 and vari.max() <= 1.0
