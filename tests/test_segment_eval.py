"""Tests for model-output handling, evaluation helpers, downloads and the Hub card."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from leaftrack.datasets.download import download, safe_extract
from leaftrack.evaluation import (
    PostProcess,
    load_gt_masks,
    load_split,
    measure_latency,
    results_markdown,
    select_setting,
    summarize_by,
)
from leaftrack.hub import model_card
from leaftrack.metrics import score_image
from leaftrack.segment import LeafInstance, drop_neighbor_leaves, to_instances, touches_border


def leaf(y0: int, y1: int, x0: int, x1: int, conf: float = 0.9, shape=(100, 100)) -> LeafInstance:
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return LeafInstance(m, conf, (x0, y0, x1, y1))


def test_touches_border() -> None:
    assert touches_border(leaf(0, 5, 40, 60).mask)
    assert not touches_border(leaf(40, 60, 40, 60).mask)


def test_neighbor_leaf_at_edge_is_dropped_but_long_own_leaf_is_kept() -> None:
    centre = leaf(40, 60, 40, 60)
    neighbor = leaf(0, 15, 0, 15)  # corner fragment, far from the plant centre
    own_long = leaf(45, 55, 50, 100)  # grows from the centre out to the right edge
    kept = drop_neighbor_leaves([centre, neighbor, own_long])
    assert neighbor not in kept
    assert centre in kept and own_long in kept


def test_neighbor_filter_uses_image_centre_when_all_leaves_touch_border() -> None:
    near_centre = leaf(0, 50, 45, 55)  # reaches the middle
    far = leaf(0, 10, 90, 100)
    kept = drop_neighbor_leaves([near_centre, far])
    assert kept == [near_centre]
    assert drop_neighbor_leaves([]) == []


def test_postprocess_thresholds_then_filters() -> None:
    insts = [leaf(40, 60, 40, 60, 0.9), leaf(30, 40, 30, 40, 0.2), leaf(0, 10, 0, 10, 0.95)]
    assert len(PostProcess(0.5, False).apply(insts)) == 2
    assert len(PostProcess(0.5, True).apply(insts)) == 1


def test_select_setting_prefers_higher_sbd() -> None:
    a = np.zeros((8, 8), bool)
    a[:4] = True
    good, bad = PostProcess(0.5, True), PostProcess(0.25, False)
    results = {
        bad: [({"rig": "x"}, score_image([a, ~a], [a], (8, 8)))],
        good: [({"rig": "x"}, score_image([a], [a], (8, 8)))],
    }
    assert select_setting(results) == good


# --- segment ---------------------------------------------------------------


def test_to_instances_sorts_by_confidence_and_thresholds_masks() -> None:
    masks = np.zeros((3, 10, 10), np.float32)
    masks[0, :2, :2] = 0.9
    masks[1, 5:, 5:] = 0.9
    masks[2, 0, 9] = 0.4  # below 0.5 everywhere -> empty -> dropped
    out = to_instances(masks, np.array([0.3, 0.8, 0.9]), np.zeros((3, 4)))
    assert [round(i.confidence, 1) for i in out] == [0.8, 0.3]
    assert out[0].area == 25 and out[0].mask.dtype == bool


def test_to_instances_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        to_instances(np.zeros((2, 4, 4)), np.zeros(1), np.zeros((2, 4)))


def test_instance_centroid() -> None:
    mask = np.zeros((10, 10), bool)
    mask[2:4, 6:8] = True
    inst = LeafInstance(mask, 0.9, (6, 2, 8, 4))
    assert inst.centroid == pytest.approx((6.5, 2.5))
    empty = LeafInstance(np.zeros((4, 4), bool), 0.5, (0, 0, 2, 4))
    assert empty.centroid == (1.0, 2.0)


# --- evaluation ------------------------------------------------------------


def test_split_rows_and_gt_masks(fake_komatsuna: Path, tmp_path: Path) -> None:
    from tests.test_pipeline import load_script

    out = tmp_path / "processed"
    load_script("convert_komatsuna").main(["--raw", str(fake_komatsuna), "--out", str(out)])
    load_script("split_dataset").main(
        ["--data", str(out), "--split-file", str(tmp_path / "s.json")]
    )
    rows = load_split(out, "test")
    assert rows and all(r["image"].startswith("images/") for r in rows)

    row = dict(rows[0])
    from_raw = load_gt_masks(row, out)
    row["raw_label"] = str(tmp_path / "missing.png")  # force the polygon fallback
    from_polygons = load_gt_masks(row, out)
    assert len(from_raw) == len(from_polygons) == int(row["n_leaves"])
    for a, b in zip(from_raw, from_polygons, strict=True):
        assert (a & b).sum() / (a | b).sum() > 0.8


def test_summarize_by_rig_and_markdown() -> None:
    a = np.zeros((8, 8), bool)
    a[:4] = True
    results = [
        ({"rig": "multi"}, score_image([a], [a], (8, 8))),
        ({"rig": "rgbd"}, score_image([], [a], (8, 8))),
    ]
    summary = summarize_by(results)
    assert set(summary) == {"all", "multi", "rgbd"}
    assert summary["multi"]["sbd"] == 1.0 and summary["rgbd"]["sbd"] == 0.0
    report = {
        "leaf_metrics": summary,
        "map": {"mask_map50": 0.9, "mask_map50_95": 0.7, "box_map50": 0.95, "box_map50_95": 0.8},
        "latency": {"median_ms": 120.0, "p90_ms": 150.0, "threads": 2, "imgsz": 640},
    }
    md = results_markdown("yolo11n", report)
    assert "| multi | 1 | 1.000" in md and "0.700" in md and "median 120 ms" in md


def test_measure_latency_uses_every_image() -> None:
    class Fake:
        calls = 0

        def predict(self, image: np.ndarray) -> list:
            Fake.calls += 1
            return []

    stats = measure_latency(Fake(), [np.zeros((4, 4, 3))] * 5, warmup=2)  # type: ignore[arg-type]
    assert stats["images"] == 5 and Fake.calls == 7
    with pytest.raises(ValueError):
        measure_latency(Fake(), [])  # type: ignore[arg-type]


# --- download --------------------------------------------------------------


def test_safe_extract_blocks_path_traversal(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_extracts_normal_archive(tmp_path: Path) -> None:
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("folder/a.txt", "hello")
    assert safe_extract(good, tmp_path / "out") == ["folder/a.txt"]
    assert (tmp_path / "out" / "folder" / "a.txt").read_text() == "hello"


def test_download_skips_complete_file(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"12345")
    # An unreachable URL proves no request is made when the size already matches.
    assert download("http://invalid.invalid/x.bin", f, expected_size=5) == f


# --- hub -------------------------------------------------------------------


def test_model_card_embeds_results(tmp_path: Path) -> None:
    a = np.ones((4, 4), bool)
    report = {"leaf_metrics": summarize_by([({"rig": "rgbd"}, score_image([a], [a], (4, 4)))])}
    card = model_card("user/leaftrack-yolo11n-seg", "leaftrack-yolo11n-seg", report)
    assert card.startswith("---\nlicense: mit")
    assert 'LeafSegmenter.from_hub("user/leaftrack-yolo11n-seg")' in card
    assert "| rgbd | 1 | 1.000" in card
    json.dumps(report)  # report stays JSON-serialisable
    assert "Not yet evaluated" in model_card("u/m", "m", None)
