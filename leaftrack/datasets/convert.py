"""Dataset-agnostic conversion of per-instance masks to YOLO-seg label lines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from leaftrack.datasets.yolo import mask_to_polygon, polygon_to_mask, polygon_to_yolo_line


@dataclass
class ConversionStats:
    """Running counters describing how faithful a conversion was.

    Attributes:
        instances: masks seen.
        written: instances written to label files.
        dropped_small: instances below the minimum area.
        dropped_degenerate: instances that produced no valid polygon.
        merged: instances whose mask had several pieces that were merged.
        ious: IoU between each written polygon (re-rasterised) and its mask.
    """

    instances: int = 0
    written: int = 0
    dropped_small: int = 0
    dropped_degenerate: int = 0
    merged: int = 0
    ious: list[float] = field(default_factory=list)

    def update(self, other: ConversionStats) -> None:
        """Add another stats object's counters into this one."""
        self.instances += other.instances
        self.written += other.written
        self.dropped_small += other.dropped_small
        self.dropped_degenerate += other.dropped_degenerate
        self.merged += other.merged
        self.ious.extend(other.ious)

    def summary(self) -> str:
        """One-paragraph human-readable report."""
        iou = np.asarray(self.ious) if self.ious else np.asarray([np.nan])
        return (
            f"instances={self.instances} written={self.written} "
            f"dropped_small={self.dropped_small} dropped_degenerate={self.dropped_degenerate} "
            f"merged_multi_piece={self.merged}\n"
            f"polygon-vs-mask IoU: mean={np.nanmean(iou):.4f} "
            f"p5={np.nanpercentile(iou, 5):.4f} min={np.nanmin(iou):.4f}"
        )


def masks_to_yolo_lines(
    masks: Mapping[int, np.ndarray],
    width: int,
    height: int,
    class_id: int = 0,
    epsilon: float = 1.0,
    min_instance_area: int = 5,
    min_component_area: int = 10,
) -> tuple[list[str], list[int], ConversionStats]:
    """Convert ``{instance_id: mask}`` to YOLO-seg lines.

    Args:
        masks: boolean / binary masks keyed by instance (e.g. leaf) ID.
        width: image width in pixels.
        height: image height in pixels.
        class_id: class index written on every line.
        epsilon: polygon simplification tolerance (pixels).
        min_instance_area: masks with fewer pixels are dropped.
        min_component_area: secondary mask pieces smaller than this are dropped.

    Returns:
        ``(lines, instance_ids, stats)``; ``instance_ids[i]`` is the ID of the
        instance written on ``lines[i]``, so ground-truth identities survive
        the conversion (needed later for tracking evaluation).
    """
    stats = ConversionStats()
    lines: list[str] = []
    ids: list[int] = []
    for instance_id in sorted(masks):
        mask = np.asarray(masks[instance_id]) > 0
        stats.instances += 1
        if int(mask.sum()) < min_instance_area:
            stats.dropped_small += 1
            continue
        result = mask_to_polygon(mask, epsilon=epsilon, min_component_area=min_component_area)
        if len(result.polygon) < 3:
            stats.dropped_degenerate += 1
            continue
        if result.n_components > 1:
            stats.merged += 1
        raster = polygon_to_mask(result.polygon, width, height).astype(bool)
        union = np.logical_or(raster, mask).sum()
        stats.ious.append(float(np.logical_and(raster, mask).sum() / union) if union else 0.0)
        lines.append(polygon_to_yolo_line(result.polygon, width, height, class_id))
        ids.append(instance_id)
        stats.written += 1
    return lines, ids, stats
