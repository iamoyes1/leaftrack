"""Draw converted YOLO-seg polygons back onto images for a visual sanity check.

Each output panel shows, left to right: the image, the converted polygons
(numbered by ground-truth leaf ID), and the original color label. Samples are
spread across rigs, plants and growth stages and include the hardest cases
(most leaves, multi-piece masks are common there).

Usage::

    python data/scripts/visualize_labels.py --data data/processed/komatsuna \
        --out outputs/check --n 6
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from leaftrack.datasets.yolo import parse_yolo_line
from leaftrack.viz import draw_polygons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed/komatsuna"))
    p.add_argument("--out", type=Path, default=Path("outputs/check"))
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--panel-height", type=int, default=480)
    return p.parse_args(argv)


def pick_samples(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """Pick ``n`` frames covering both rigs, different plants and growth stages.

    Half the picks are early/mid/late frames from alternating rigs, the rest
    are the frames with the most leaves (the hardest to convert correctly).
    """
    rng = random.Random(seed)
    by_rig: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_rig.setdefault(r["rig"], []).append(r)
    rigs = sorted(by_rig)
    picks: list[dict[str, str]] = []
    stages = [(0, 19), (20, 39), (40, 59)]
    for k in range(n - n // 2):
        rig = rigs[k % len(rigs)]
        lo, hi = stages[k % len(stages)]
        candidates = [r for r in by_rig[rig] if r not in picks]
        staged = [r for r in candidates if lo <= int(r["frame_index"]) <= hi]
        if staged or candidates:
            picks.append(rng.choice(staged or candidates))
    crowded = sorted(rows, key=lambda r: (-int(r["n_leaves"]), r["stem"]))
    used_sequences = {r["sequence_id"] for r in picks}
    while len(picks) < n:
        # Take the most crowded unused plant from the least represented rig.
        counts = {rig: sum(p["rig"] == rig for p in picks) for rig in rigs}
        candidates = [r for r in crowded if r not in picks]
        fresh = [r for r in candidates if r["sequence_id"] not in used_sequences] or candidates
        if not fresh:
            break
        best = min(fresh, key=lambda r: (counts[r["rig"]], -int(r["n_leaves"]), r["stem"]))
        picks.append(best)
        used_sequences.add(best["sequence_id"])
    return picks


def render_panel(row: dict[str, str], data: Path, panel_height: int) -> np.ndarray:
    """Build the image | polygons | original-label panel for one frame."""
    image = np.asarray(Image.open(data / row["image"]).convert("RGB"))
    raw_label = np.asarray(Image.open(row["raw_label"]).convert("RGB"))
    h, w = image.shape[:2]
    lines = (data / row["label"]).read_text(encoding="utf-8").split("\n")
    polygons = [parse_yolo_line(line, w, h)[1] for line in lines if line.strip()]
    ids = [int(x) for x in row["leaf_ids"].split(";")] if row["leaf_ids"] else []

    scale = panel_height / h
    size = (round(w * scale), panel_height)
    big_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    big_label = cv2.resize(raw_label, size, interpolation=cv2.INTER_NEAREST)
    overlay = draw_polygons(big_image, [p * scale for p in polygons], ids)

    gap = np.full((panel_height, 6, 3), 255, dtype=np.uint8)
    panel = np.hstack([big_image, gap, overlay, gap, big_label])
    header = np.full((28, panel.shape[1], 3), 255, dtype=np.uint8)
    title = (
        f"{row['stem']}  |  {row['n_leaves']} leaves  |  "
        "image / converted polygons / original label"
    )
    cv2.putText(header, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, panel])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = args.data / "manifest.csv"
    if not manifest.exists():
        print(f"{manifest} not found; run convert_komatsuna.py first", file=sys.stderr)
        return 1
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    args.out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(pick_samples(rows, args.n, args.seed), 1):
        path = args.out / f"{i:02d}_{row['stem']}.png"
        Image.fromarray(render_panel(row, args.data, args.panel_height)).save(path)
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
