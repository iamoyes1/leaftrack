"""Tests for the KOMATSUNA reader and the mask -> YOLO line conversion."""

from pathlib import Path

import numpy as np
import pytest

from leaftrack.datasets.convert import masks_to_yolo_lines
from leaftrack.datasets.komatsuna import (
    LEAF_PALETTE,
    decode_label,
    discover_frames,
    parse_image_name,
)
from tests.conftest import make_label


def test_parse_multi_view_name() -> None:
    assert parse_image_name("rgb_01_03_007_05.png", "multi") == {
        "camera": 1, "plant": 3, "day": 7, "time": 5,
    }  # fmt: skip


def test_parse_rgbd_name() -> None:
    assert parse_image_name("rgb_04_009_02.png", "rgbd") == {
        "camera": None, "plant": 4, "day": 9, "time": 2,
    }  # fmt: skip


@pytest.mark.parametrize(
    "name,rig",
    [
        ("rgb_04_009_02.png", "multi"),
        ("rgb_01_03_007_05.png", "rgbd"),
        ("label_00_000_00.png", "rgbd"),
    ],
)
def test_parse_rejects_other_patterns(name: str, rig: str) -> None:
    assert parse_image_name(name, rig) is None


def test_discover_pairs_images_with_labels_only(fake_komatsuna: Path) -> None:
    frames = list(discover_frames(fake_komatsuna))
    assert len(frames) == 8 + 4  # the unlabeled rgbd image and notes.txt are skipped
    assert {f.rig for f in frames} == {"multi", "rgbd"}
    assert all(f.label_path.exists() for f in frames)
    assert len({f.stem for f in frames}) == len(frames), "stems must be unique"


def test_frame_identity_and_time(fake_komatsuna: Path) -> None:
    frame = next(f for f in discover_frames(fake_komatsuna) if f.rig == "multi" and f.camera == 1)
    assert frame.sequence_id == f"multi_p{frame.plant:02d}"
    assert frame.stem.startswith("komatsuna_multi_c01_")
    other_cam = next(
        f for f in discover_frames(fake_komatsuna)
        if f.rig == "multi" and f.camera == 0 and f.plant == frame.plant
    )  # fmt: skip
    assert other_cam.sequence_id == frame.sequence_id, "cameras of one plant share a sequence"


def test_frame_index_and_hours() -> None:
    from leaftrack.datasets.komatsuna import KomatsunaFrame

    f = KomatsunaFrame("rgbd", None, 2, day=3, time=4, image_path=Path("a"), label_path=Path("b"))
    assert f.frame_index == 22
    assert f.hours == 88


def test_decode_label_maps_palette_colors_and_ignores_noise() -> None:
    label = make_label(48, 48, [1, 3, 8], noise=True)
    masks = decode_label(label)
    assert list(masks) == [1, 3, 8]
    for leaf_id, mask in masks.items():
        assert mask.dtype == bool
        assert np.all(label[mask] == LEAF_PALETTE[leaf_id])
    union = np.logical_or.reduce(list(masks.values()))
    assert not union[46, 46], "white noise pixel must not belong to any leaf"


def test_decode_label_rejects_grayscale() -> None:
    with pytest.raises(ValueError):
        decode_label(np.zeros((4, 4), np.uint8))


def test_masks_to_yolo_lines_keeps_leaf_ids_in_line_order() -> None:
    masks = decode_label(make_label(48, 48, [2, 5, 6]))
    lines, ids, stats = masks_to_yolo_lines(masks, 48, 48)
    assert ids == [2, 5, 6]
    assert len(lines) == 3 and all(line.startswith("0 ") for line in lines)
    assert stats.written == 3 and stats.instances == 3
    assert min(stats.ious) > 0.9


def test_masks_to_yolo_lines_drops_tiny_instances() -> None:
    big = np.zeros((20, 20), bool)
    big[2:12, 2:12] = True
    tiny = np.zeros((20, 20), bool)
    tiny[15, 15] = True
    lines, ids, stats = masks_to_yolo_lines({1: big, 2: tiny}, 20, 20, min_instance_area=5)
    assert ids == [1]
    assert stats.dropped_small == 1
