"""Tests for mask <-> YOLO-seg polygon conversion."""

import numpy as np
import pytest

from leaftrack.datasets.yolo import (
    mask_to_polygon,
    parse_yolo_line,
    polygon_to_mask,
    polygon_to_yolo_line,
)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / (a | b).sum())


def test_rectangle_roundtrip_is_near_exact() -> None:
    mask = np.zeros((100, 120), np.uint8)
    mask[20:60, 30:90] = 1
    result = mask_to_polygon(mask)
    assert result.n_components == 1
    assert result.kept_fraction == pytest.approx(1.0)
    assert iou(polygon_to_mask(result.polygon, 120, 100), mask) > 0.95


def test_empty_mask_returns_empty_polygon() -> None:
    result = mask_to_polygon(np.zeros((10, 10), np.uint8))
    assert result.polygon.shape == (0, 2)
    assert result.n_components == 0


def test_multi_piece_mask_is_merged_into_one_polygon() -> None:
    mask = np.zeros((100, 100), np.uint8)
    mask[10:40, 10:40] = 1  # leaf blade
    mask[60:90, 55:85] = 1  # piece separated by an occluding leaf
    result = mask_to_polygon(mask)
    assert result.n_components == 2
    assert result.kept_fraction == pytest.approx(1.0)
    raster = polygon_to_mask(result.polygon, 100, 100)
    assert raster[25, 25] and raster[75, 70], "both pieces must be covered"
    assert iou(raster, mask) > 0.9


def test_small_secondary_specks_are_dropped_but_largest_piece_is_kept() -> None:
    mask = np.zeros((50, 50), np.uint8)
    mask[10:30, 10:30] = 1
    mask[45, 45] = 1  # one-pixel speck
    result = mask_to_polygon(mask, min_component_area=10)
    assert result.n_components == 2
    assert result.kept_fraction == pytest.approx(400 / 401)


def test_tiny_single_leaf_is_not_dropped() -> None:
    mask = np.zeros((50, 50), np.uint8)
    mask[20:23, 20:23] = 1  # 9 px: smaller than min_component_area
    result = mask_to_polygon(mask, min_component_area=10)
    assert len(result.polygon) >= 3


def test_yolo_line_is_normalised_and_parses_back() -> None:
    poly = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=float)
    line = polygon_to_yolo_line(poly, width=200, height=100, class_id=0)
    values = [float(v) for v in line.split()[1:]]
    assert line.split()[0] == "0"
    assert min(values) >= 0.0 and max(values) <= 1.0
    cls, back = parse_yolo_line(line, 200, 100)
    assert cls == 0
    np.testing.assert_allclose(back, poly, atol=1e-3)


def test_yolo_line_clips_out_of_bounds_points() -> None:
    poly = np.array([[-5, -5], [110, 0], [50, 120]], dtype=float)
    values = [float(v) for v in polygon_to_yolo_line(poly, 100, 100).split()[1:]]
    assert min(values) == 0.0 and max(values) == 1.0


def test_yolo_line_rejects_degenerate_polygon() -> None:
    with pytest.raises(ValueError):
        polygon_to_yolo_line(np.array([[0, 0], [1, 1]]), 10, 10)


def test_parse_rejects_malformed_line() -> None:
    with pytest.raises(ValueError):
        parse_yolo_line("0 0.1 0.2 0.3", 10, 10)
