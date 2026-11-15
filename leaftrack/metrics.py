"""Leaf segmentation metrics from the CVPPP leaf segmentation challenge.

Reported alongside the Ultralytics mask mAP:

* **SBD** (Symmetric Best Dice): for every leaf in one set, take its best Dice
  match in the other set and average; SBD is the smaller of the two
  directions. It rewards getting each individual leaf right.
* **FBD** (Foreground-Background Dice): Dice of the union of all leaves, i.e.
  how well the plant as a whole is separated from the background.
* **DiC** / **|DiC|** (Difference in Count): predicted minus true leaf count,
  signed and absolute. Leaf count is a key phenotyping trait.

Reference: Scharr et al., "Leaf segmentation in plant phenotyping: a
collation study", Machine Vision and Applications, 2016.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient of two boolean masks (1.0 when both are empty)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * int(np.logical_and(a, b).sum()) / total


def pairwise_dice(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """``(len(a), len(b))`` matrix of Dice scores between two mask lists."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    fa = np.stack([np.asarray(m, bool).ravel() for m in a]).astype(np.float32)
    fb = np.stack([np.asarray(m, bool).ravel() for m in b]).astype(np.float32)
    inter = fa @ fb.T
    sizes = fa.sum(1)[:, None] + fb.sum(1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sizes > 0, 2.0 * inter / sizes, 1.0)


def best_dice(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    """Mean over masks in ``a`` of their best Dice against any mask in ``b``.

    Conventions: both empty -> 1.0; exactly one empty -> 0.0.
    """
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(pairwise_dice(a, b).max(axis=1).mean())


def symmetric_best_dice(pred: Sequence[np.ndarray], gt: Sequence[np.ndarray]) -> float:
    """SBD = min(BD(pred, gt), BD(gt, pred))."""
    return min(best_dice(pred, gt), best_dice(gt, pred))


def foreground_dice(pred: Sequence[np.ndarray], gt: Sequence[np.ndarray], shape: tuple) -> float:
    """Dice between the union of predicted and the union of true leaf masks."""

    def union(masks: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros(shape, bool)
        for m in masks:
            out |= np.asarray(m, bool)
        return out

    return dice(union(pred), union(gt))


@dataclass(frozen=True)
class ImageScore:
    """CVPPP metrics for one image."""

    sbd: float
    fbd: float
    dic: int
    n_pred: int
    n_gt: int

    @property
    def abs_dic(self) -> int:
        return abs(self.dic)


def score_image(pred: Sequence[np.ndarray], gt: Sequence[np.ndarray], shape: tuple) -> ImageScore:
    """Compute all per-image metrics for predicted vs ground-truth leaf masks."""
    return ImageScore(
        sbd=symmetric_best_dice(pred, gt),
        fbd=foreground_dice(pred, gt, shape),
        dic=len(pred) - len(gt),
        n_pred=len(pred),
        n_gt=len(gt),
    )


@dataclass(frozen=True)
class Summary:
    """Dataset-level aggregate of :class:`ImageScore` values (mean and std)."""

    images: int
    sbd: float
    sbd_std: float
    fbd: float
    fbd_std: float
    dic: float
    abs_dic: float
    abs_dic_std: float
    count_accuracy: float  # fraction of images with exactly the right leaf count

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize(scores: Iterable[ImageScore]) -> Summary:
    """Aggregate per-image scores.

    Raises:
        ValueError: if ``scores`` is empty.
    """
    scores = list(scores)
    if not scores:
        raise ValueError("No scores to summarize")
    sbd = np.array([s.sbd for s in scores])
    fbd = np.array([s.fbd for s in scores])
    dic = np.array([s.dic for s in scores], dtype=float)
    return Summary(
        images=len(scores),
        sbd=float(sbd.mean()),
        sbd_std=float(sbd.std()),
        fbd=float(fbd.mean()),
        fbd_std=float(fbd.std()),
        dic=float(dic.mean()),
        abs_dic=float(np.abs(dic).mean()),
        abs_dic_std=float(np.abs(dic).std()),
        count_accuracy=float((dic == 0).mean()),
    )
