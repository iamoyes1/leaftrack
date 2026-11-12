"""Leaf instance segmentation (stub, Phase 2).

Will wrap an Ultralytics YOLO11n-seg / YOLO11s-seg model behind a small,
framework-agnostic interface so the rest of the package never imports
Ultralytics directly.

Planned API:

- ``LeafSegmenter(weights, device="cpu", conf=0.25, imgsz=640)``: loads weights
  from a local path or the Hugging Face Hub.
- ``LeafSegmenter.predict(image) -> list[LeafInstance]``: returns one entry per
  detected leaf with a binary mask, polygon, bounding box, confidence and
  pixel area.

CPU inference speed is a hard requirement (Hugging Face Spaces free tier), so
the default model will be the nano variant at a modest input size.
"""
