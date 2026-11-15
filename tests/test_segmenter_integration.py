"""Integration test for LeafSegmenter with a real Ultralytics model.

Skipped unless Ultralytics is installed and weights are present locally, so the
default test run stays fast and offline. Set LEAFTRACK_TEST_WEIGHTS to test a
specific checkpoint (e.g. a trained best.pt).
"""

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("LEAFTRACK_TEST_WEIGHTS", ROOT / "yolo11n-seg.pt"))


@pytest.mark.skipif(not WEIGHTS.exists(), reason=f"weights not found: {WEIGHTS}")
def test_predict_returns_full_resolution_instances() -> None:
    pytest.importorskip("ultralytics")
    from leaftrack.segment import LeafSegmenter

    segmenter = LeafSegmenter(WEIGHTS, device="cpu", conf=0.01, imgsz=320)
    image = np.random.default_rng(0).integers(0, 255, (190, 166, 3), dtype=np.uint8)
    instances = segmenter.predict(image)
    assert isinstance(instances, list)
    confidences = [inst.confidence for inst in instances]
    assert confidences == sorted(confidences, reverse=True)
    for inst in instances:
        assert inst.mask.shape == (190, 166) and inst.mask.dtype == bool
        assert 0.0 <= inst.confidence <= 1.0

    with pytest.raises(ValueError):
        segmenter.predict(np.zeros((10, 10), np.uint8))
