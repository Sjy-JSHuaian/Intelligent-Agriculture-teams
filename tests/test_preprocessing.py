"""预处理与切片模块的单元测试（不依赖 LoveDA 数据，使用合成数组）。"""
import numpy as np
import pytest

from datasets.preprocessing import (
    compute_exg, compute_gli, compute_ndi, compute_vari,
    standardize_rgb, to_float01,
)
from datasets.tiling import tile_array, tile_pair, tile_positions
from datasets.dataset_loader import IGNORE_INDEX, NUM_CLASSES, RAW_CLASSES, remap_mask


def test_loveda_class_mapping():
    # 官方 README：0=no-data(忽略), 1=background, ..., 7=agriculture
    assert RAW_CLASSES[0] == "no-data"
    assert RAW_CLASSES[7] == "agriculture"
    assert NUM_CLASSES == 7


def test_remap_mask():
    raw = np.array([[0, 1, 7], [3, 5, 6]], dtype=np.uint8)
    expected = np.array([[IGNORE_INDEX, 0, 6], [2, 4, 5]], dtype=np.int64)
    assert (remap_mask(raw) == expected).all()


def test_standardize_rgb_shape():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = standardize_rgb(img)
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.float32


def test_to_float01_range():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert to_float01(img).max() == 0.0
    img[:] = 255
    assert to_float01(img).min() == 1.0


def test_exg_pure_green_positive():
    # 纯绿像素 (R=0, G=1, B=0) -> ExG = 2
    green = np.zeros((1, 1, 3), dtype=np.float32)
    green[..., 1] = 1.0
    assert compute_exg(green)[0, 0] == pytest.approx(2.0)


def test_vari_no_nan():
    # 全零像素不应产生 NaN（分母加了 epsilon）
    black = np.zeros((8, 8, 3), dtype=np.float32)
    assert np.isfinite(compute_vari(black)).all()
    assert np.isfinite(compute_gli(black)).all()
    assert np.isfinite(compute_ndi(black)).all()


def test_tile_positions_cover():
    positions = tile_positions(1024, 1024, 256, 256)
    assert len(positions) == 16  # 4x4
    assert (0, 0) in positions


def test_tile_pair_filters_background():
    mask = np.zeros((512, 512), dtype=np.uint8)          # 全背景
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    tiles = tile_pair(image, mask, 256, 256, min_valid_ratio=0.05, ignore_index=0)
    assert len(tiles) == 0  # 无有效像素，全部丢弃

    mask[10:200, 10:200] = 7  # 仅在左上角 256x256 切片内的农田(agriculture)区域
    tiles = tile_pair(image, mask, 256, 256, min_valid_ratio=0.05, ignore_index=0)
    assert len(tiles) == 1   # 只有左上角切片被保留


def test_tile_array_size():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tiles = tile_array(img, 32, 32)
    assert len(tiles) > 0
    assert all(t.shape == (32, 32, 3) for t in tiles)
