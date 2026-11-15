"""Drawing helpers for leaf instances (used by data checks and, later, the demo)."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import cv2
import numpy as np


def id_color(index: int) -> tuple[int, int, int]:
    """Return a distinct, deterministic RGB color for an instance or track ID.

    Uses golden-ratio hue stepping, so neighbouring IDs get well-separated hues.
    """
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_polygons(
    image: np.ndarray,
    polygons: Sequence[np.ndarray],
    ids: Sequence[int] | None = None,
    alpha: float = 0.45,
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """Overlay filled, outlined polygons on an RGB image.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image (not modified).
        polygons: pixel-space ``(N, 2)`` arrays, one per instance.
        ids: optional IDs used for colors and text labels; defaults to
            ``1..len(polygons)``.
        alpha: fill opacity.
        thickness: outline width in pixels.
        show_labels: draw each ID at its polygon's centroid.

    Returns:
        A new RGB uint8 image with the overlay.
    """
    ids = list(ids) if ids is not None else list(range(1, len(polygons) + 1))
    if len(ids) != len(polygons):
        raise ValueError("ids and polygons must have the same length")

    base = np.ascontiguousarray(image, dtype=np.uint8).copy()
    fill = base.copy()
    int_polys = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for p in polygons]
    for poly, idx in zip(int_polys, ids, strict=True):
        cv2.fillPoly(fill, [poly], id_color(idx))
    out = cv2.addWeighted(fill, alpha, base, 1 - alpha, 0)
    for poly, idx in zip(int_polys, ids, strict=True):
        cv2.polylines(out, [poly], isClosed=True, color=id_color(idx), thickness=thickness)

    if show_labels:
        scale = max(0.35, min(out.shape[:2]) / 900)
        for poly, idx in zip(int_polys, ids, strict=True):
            cx, cy = poly.reshape(-1, 2).mean(axis=0).astype(int)
            text = str(idx)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            org = (int(cx - tw / 2), int(cy + th / 2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(out, text, org, font, scale, (0, 0, 0), 3, cv2.LINE_AA)  # outline
            cv2.putText(out, text, org, font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out
