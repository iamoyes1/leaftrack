"""Leaf instance segmentation with a YOLO11-seg model.

:class:`LeafSegmenter` hides Ultralytics behind a small interface: an RGB
image goes in, a list of :class:`LeafInstance` comes out, each with a
full-resolution mask. The rest of the package (tracking, analytics, demo)
never imports Ultralytics directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(eq=False)  # holds arrays: compare instances by identity
class LeafInstance:
    """One detected leaf.

    Attributes:
        mask: ``(H, W)`` boolean mask at the input image's resolution.
        confidence: detection score in ``[0, 1]``.
        box: ``(x1, y1, x2, y2)`` pixel bounding box.
        polygon: ``(N, 2)`` pixel outline, may be empty for tiny masks.
    """

    mask: np.ndarray
    confidence: float
    box: tuple[float, float, float, float]
    polygon: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))

    @property
    def area(self) -> int:
        """Leaf area in pixels."""
        return int(self.mask.sum())

    @property
    def centroid(self) -> tuple[float, float]:
        """Mask centroid ``(x, y)`` in pixels (box centre for an empty mask)."""
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            x1, y1, x2, y2 = self.box
            return (x1 + x2) / 2, (y1 + y2) / 2
        return float(xs.mean()), float(ys.mean())


def to_instances(
    masks: np.ndarray,
    confidences: np.ndarray,
    boxes: np.ndarray,
    polygons: list[np.ndarray] | None = None,
    min_area: int = 1,
) -> list[LeafInstance]:
    """Build instances from raw model outputs, sorted by confidence.

    Args:
        masks: ``(N, H, W)`` array; values above 0.5 count as foreground.
        confidences: ``(N,)`` scores.
        boxes: ``(N, 4)`` ``xyxy`` boxes.
        polygons: optional ``N`` pixel outlines.
        min_area: masks with fewer foreground pixels are discarded.

    Raises:
        ValueError: if the array lengths disagree.
    """
    masks = np.asarray(masks)
    n = len(masks)
    if len(confidences) != n or len(boxes) != n or (polygons is not None and len(polygons) != n):
        raise ValueError("masks, confidences, boxes and polygons must have equal length")
    out = []
    for i in np.argsort(-np.asarray(confidences, dtype=float), kind="stable"):
        mask = masks[i] > 0.5
        if int(mask.sum()) < min_area:
            continue
        poly = (
            np.asarray(polygons[i], np.float32).reshape(-1, 2)
            if polygons is not None
            else np.empty((0, 2), np.float32)
        )
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        out.append(LeafInstance(mask, float(confidences[i]), (x1, y1, x2, y2), poly))
    return out


def touches_border(mask: np.ndarray, margin: int = 1) -> bool:
    """Whether any foreground pixel lies within ``margin`` pixels of the edge."""
    m = max(1, margin)
    return bool(mask[:m].any() or mask[-m:].any() or mask[:, :m].any() or mask[:, -m:].any())


def drop_neighbor_leaves(
    instances: list[LeafInstance], radius_frac: float = 0.25
) -> list[LeafInstance]:
    """Remove leaves that belong to neighbouring plants cut off by the frame.

    Plant crops from a tray often contain parts of adjacent plants at the
    edges. A leaf of the plant of interest grows out from the plant centre, so
    even when it reaches the image edge, part of it comes close to that
    centre. A neighbour's leaf touches the edge and stays near it.

    The plant centre is the area-weighted centroid of all leaves that do not
    touch the border (the image centre if there are none). A border-touching
    leaf is kept only if some pixel lies within ``radius_frac * min(H, W)`` of
    that centre.
    """
    if not instances:
        return []
    h, w = instances[0].mask.shape
    # Predicted masks often stop a pixel or two short of the frame after
    # upsampling, so "touching" allows a small margin.
    margin = max(2, round(0.01 * min(h, w)))
    edge = {id(i): touches_border(i.mask, margin) for i in instances}
    inner = [i for i in instances if not edge[id(i)]]
    if inner:
        areas = np.array([i.area for i in inner], dtype=float)
        cents = np.array([i.centroid for i in inner])
        cx, cy = (cents * areas[:, None]).sum(0) / areas.sum()
    else:
        cx, cy = w / 2, h / 2
    radius = radius_frac * min(h, w)
    kept = []
    for inst in instances:
        if not edge[id(inst)]:
            kept.append(inst)
            continue
        ys, xs = np.nonzero(inst.mask)
        if np.min((xs - cx) ** 2 + (ys - cy) ** 2) <= radius**2:
            kept.append(inst)
    return kept


class LeafSegmenter:
    """YOLO11-seg leaf segmenter.

    Args:
        weights: local ``.pt`` path (use :meth:`from_hub` for the Hub).
        device: ``"cpu"``, ``"0"`` (first GPU), etc.
        conf: minimum detection confidence.
        iou: NMS IoU threshold.
        imgsz: inference size; smaller is faster on CPU.
        max_det: maximum leaves per image.
        drop_neighbors: remove leaves of adjacent plants cut off by the image
            edge (see :func:`drop_neighbor_leaves`).
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        max_det: int = 50,
        drop_neighbors: bool = True,
    ) -> None:
        from ultralytics import YOLO  # lazy: keeps `import leaftrack` light

        self.weights = str(weights)
        self.model = YOLO(self.weights, task="segment")
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.drop_neighbors = drop_neighbors

    @classmethod
    def from_hub(cls, repo_id: str, filename: str = "best.pt", **kwargs: Any) -> LeafSegmenter:
        """Download weights from the Hugging Face Hub (cached) and load them."""
        from huggingface_hub import hf_hub_download

        return cls(hf_hub_download(repo_id=repo_id, filename=filename), **kwargs)

    def predict(self, image: np.ndarray) -> list[LeafInstance]:
        """Segment the leaves in one ``(H, W, 3)`` uint8 **RGB** image."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")
        bgr = np.ascontiguousarray(image[..., ::-1])  # Ultralytics treats arrays as BGR
        result = self.model.predict(
            bgr,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            retina_masks=True,  # masks at the original image resolution
            verbose=False,
        )[0]
        if result.masks is None or len(result.masks) == 0:
            return []
        instances = to_instances(
            masks=result.masks.data.cpu().numpy(),
            confidences=result.boxes.conf.cpu().numpy(),
            boxes=result.boxes.xyxy.cpu().numpy(),
            polygons=list(result.masks.xy),
        )
        return drop_neighbor_leaves(instances) if self.drop_neighbors else instances
