"""Conversion between binary instance masks and YOLO-seg polygon labels.

A YOLO-seg label file holds one line per instance::

    <class_id> x1 y1 x2 y2 ... xn yn

with every coordinate normalised to ``[0, 1]`` by the image width / height.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

Polygon = np.ndarray  # float array of shape (N, 2) in pixel (x, y) coordinates


@dataclass(frozen=True)
class PolygonResult:
    """A single-instance polygon plus how much of the mask it represents.

    Attributes:
        polygon: ``(N, 2)`` float array of pixel coordinates, or empty if the
            mask produced no usable contour.
        kept_fraction: area of the kept connected component(s) divided by the
            full mask area, so callers can report fragmented instances.
        n_components: number of connected components found in the mask.
    """

    polygon: Polygon
    kept_fraction: float
    n_components: int


def mask_to_polygon(
    mask: np.ndarray,
    epsilon: float = 1.0,
    min_component_area: int = 10,
) -> PolygonResult:
    """Convert one binary instance mask to a single YOLO-compatible polygon.

    Leaves are sometimes split into several pieces, e.g. when an overlapping
    leaf hides part of a petiole. YOLO-seg stores one polygon per instance, so
    the pieces are joined into one outline by connecting each piece to the
    outline through its closest pair of points (a zero-width "bridge", the
    same approach Ultralytics uses when converting COCO).

    The largest piece is always kept, however small, so tiny new leaves are
    not lost; additional pieces smaller than ``min_component_area`` pixels are
    treated as annotation specks and dropped.

    Args:
        mask: 2-D array; non-zero pixels belong to the instance.
        epsilon: Douglas-Peucker simplification tolerance in pixels. ``0``
            disables simplification.
        min_component_area: minimum pixel area for secondary pieces.

    Returns:
        A :class:`PolygonResult`. ``polygon`` is empty when no piece yields a
        polygon with at least three vertices (e.g. an all-zero mask).
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    total_area = int(binary.sum())
    if total_area == 0:
        return PolygonResult(np.empty((0, 2), dtype=np.float32), 0.0, 0)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    n_components = n_labels - 1
    order = sorted(range(1, n_labels), key=lambda k: -stats[k, cv2.CC_STAT_AREA])

    pieces: list[Polygon] = []
    kept_area = 0
    for rank, k in enumerate(order):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if rank > 0 and area < min_component_area:
            break  # sorted by area, so every remaining piece is smaller too
        piece = (labels == k).astype(np.uint8)
        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=len)
        if epsilon > 0:
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(simplified) >= 3:
                contour = simplified
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue
        pieces.append(pts)
        kept_area += area

    if not pieces:
        return PolygonResult(np.empty((0, 2), dtype=np.float32), 0.0, n_components)
    return PolygonResult(_merge_polygons(pieces), kept_area / total_area, n_components)


def _merge_polygons(polygons: Sequence[Polygon]) -> Polygon:
    """Join several closed polygons into one outline via closest-point bridges.

    Each polygon after the first is rotated to start at the vertex nearest to
    the current outline, inserted at that nearest outline vertex, and closed
    back to it, so the result traces every input polygon exactly once.
    """
    merged = np.asarray(polygons[0], dtype=np.float32)
    for poly in polygons[1:]:
        poly = np.asarray(poly, dtype=np.float32)
        d = np.linalg.norm(merged[:, None, :] - poly[None, :, :], axis=-1)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        rotated = np.roll(poly, -j, axis=0)
        loop = np.vstack([rotated, rotated[:1]])  # walk the piece and return to its start
        merged = np.vstack([merged[: i + 1], loop, merged[i:]])
    return merged


def polygon_to_yolo_line(polygon: Polygon, width: int, height: int, class_id: int = 0) -> str:
    """Format a pixel-space polygon as one normalised YOLO-seg label line.

    Args:
        polygon: ``(N, 2)`` array of ``(x, y)`` pixel coordinates, ``N >= 3``.
        width: image width in pixels.
        height: image height in pixels.
        class_id: integer class index (LeafTrack uses a single class, ``0``).

    Raises:
        ValueError: if the polygon has fewer than three points.
    """
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError(f"A polygon needs at least 3 points, got {len(pts)}.")
    norm = pts / np.array([width, height], dtype=np.float64)
    norm = np.clip(norm, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
    return f"{class_id} {coords}"


def parse_yolo_line(line: str, width: int, height: int) -> tuple[int, Polygon]:
    """Parse one YOLO-seg label line back into ``(class_id, pixel_polygon)``."""
    parts = line.split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError(f"Malformed YOLO-seg line: {line!r}")
    class_id = int(parts[0])
    pts = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
    return class_id, pts * np.array([width, height], dtype=np.float64)


def polygon_to_mask(polygon: Polygon, width: int, height: int) -> np.ndarray:
    """Rasterise a pixel-space polygon to a ``uint8`` mask (used in checks/tests)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(np.asarray(polygon)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask
