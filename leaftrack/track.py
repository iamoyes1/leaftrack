"""Leaf tracking across time-lapse frames (stub, Phase 3).

Will assign stable leaf IDs across consecutive frames of a single plant.
Leaves barely move between frames but grow, appear and are occasionally
occluded, so the default tracker will be a lightweight, CPU-only
mask-IoU + centroid-distance matcher solved with the Hungarian algorithm,
with a short memory for leaves that disappear for a few frames.

Planned API:

- ``LeafTracker(iou_threshold=0.3, max_missed=3)``
- ``LeafTracker.update(instances) -> list[TrackedLeaf]``: called once per
  frame; returns the instances annotated with persistent ``track_id`` values.
- ``LeafTracker.reset()``: start a new sequence.
"""
