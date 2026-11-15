"""Tests for CVPPP leaf segmentation metrics."""

import numpy as np
import pytest

from leaftrack.metrics import (
    best_dice,
    dice,
    foreground_dice,
    pairwise_dice,
    score_image,
    summarize,
    symmetric_best_dice,
)


def rect(y0: int, y1: int, x0: int, x1: int, shape: tuple = (20, 20)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


A, B, C = rect(0, 5, 0, 5), rect(10, 15, 10, 15), rect(0, 5, 10, 15)


def test_dice_basic_cases() -> None:
    assert dice(A, A) == 1.0
    assert dice(A, B) == 0.0
    assert dice(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0
    half = rect(0, 5, 0, 10)  # contains A, twice its size
    assert dice(A, half) == pytest.approx(2 * 25 / (25 + 50))


def test_pairwise_matches_scalar_dice() -> None:
    preds, gts = [A, B, rect(0, 5, 0, 10)], [A, C]
    matrix = pairwise_dice(preds, gts)
    assert matrix.shape == (3, 2)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            assert matrix[i, j] == pytest.approx(dice(p, g))


def test_perfect_prediction_scores_one() -> None:
    s = score_image([A, B], [B, A], (20, 20))
    assert s.sbd == pytest.approx(1.0) and s.fbd == pytest.approx(1.0) and s.dic == 0


def test_sbd_penalises_missed_leaf_but_not_in_one_direction() -> None:
    pred, gt = [A], [A, B]
    assert best_dice(pred, gt) == pytest.approx(1.0)  # every prediction is perfect...
    assert best_dice(gt, pred) == pytest.approx(0.5)  # ...but half the leaves are missed
    assert symmetric_best_dice(pred, gt) == pytest.approx(0.5)


def test_merged_prediction_lowers_sbd_and_count() -> None:
    merged = A | B
    s = score_image([merged], [A, B], (20, 20))
    assert s.dic == -1 and s.abs_dic == 1
    assert s.fbd == pytest.approx(1.0)  # foreground is still perfect
    assert s.sbd < 1.0


def test_empty_conventions() -> None:
    assert symmetric_best_dice([], []) == 1.0
    assert symmetric_best_dice([A], []) == 0.0
    assert foreground_dice([], [], (4, 4)) == 1.0


def test_summarize_aggregates() -> None:
    scores = [score_image([A, B], [A, B], (20, 20)), score_image([A], [A, B], (20, 20))]
    s = summarize(scores)
    assert s.images == 2
    assert s.abs_dic == pytest.approx(0.5) and s.dic == pytest.approx(-0.5)
    assert s.count_accuracy == pytest.approx(0.5)
    assert s.sbd == pytest.approx((1.0 + 0.5) / 2)


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize([])
