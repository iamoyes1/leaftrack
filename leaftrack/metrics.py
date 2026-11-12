"""Evaluation metrics (stub, Phases 2-3).

Will implement the metrics reported in the README results table:

- Segmentation: mask mAP50 / mAP50-95 (from Ultralytics validation), plus the
  CVPPP leaf-segmentation metrics Symmetric Best Dice (SBD) and absolute
  Difference in Count (|DiC|).
- Tracking: ID switches and a MOTA-style score computed against ground-truth
  leaf IDs on time-lapse sequences, where those labels exist.
"""
