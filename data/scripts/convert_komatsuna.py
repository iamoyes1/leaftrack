"""Convert KOMATSUNA color-coded leaf labels to YOLO-seg polygon labels.

Usage::

    python data/scripts/convert_komatsuna.py \
        --raw data/raw/komatsuna --out data/processed/komatsuna

Output layout::

    <out>/images/<stem>.png     copied plant crops
    <out>/labels/<stem>.txt     one YOLO-seg polygon per leaf (class 0 = leaf)
    <out>/manifest.csv          one row per frame: rig, camera, plant, day,
                                time, size, and the ground-truth leaf ID of
                                every label line (in line order)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from leaftrack.datasets.convert import ConversionStats, masks_to_yolo_lines
from leaftrack.datasets.komatsuna import decode_label, discover_frames

MANIFEST_FIELDS = [
    "stem", "rig", "camera", "plant", "sequence_id", "day", "time", "frame_index",
    "hours", "width", "height", "n_leaves", "leaf_ids", "image", "label", "raw_label",
]  # fmt: skip


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, default=Path("data/raw/komatsuna"))
    p.add_argument("--out", type=Path, default=Path("data/processed/komatsuna"))
    p.add_argument("--epsilon", type=float, default=1.0, help="polygon simplification (px)")
    p.add_argument("--min-instance-area", type=int, default=5, help="drop smaller leaves (px)")
    p.add_argument("--min-component-area", type=int, default=10, help="drop smaller specks (px)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames = list(discover_frames(args.raw))
    if not frames:
        print(f"No KOMATSUNA frames found under {args.raw}", file=sys.stderr)
        return 1

    img_out, lbl_out = args.out / "images", args.out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total = ConversionStats()
    rows = []
    for i, frame in enumerate(frames, 1):
        label = np.asarray(Image.open(frame.label_path).convert("RGB"))
        height, width = label.shape[:2]
        with Image.open(frame.image_path) as im:
            if im.size != (width, height):
                raise ValueError(f"Size mismatch: {frame.image_path} vs {frame.label_path}")

        lines, leaf_ids, stats = masks_to_yolo_lines(
            decode_label(label),
            width,
            height,
            epsilon=args.epsilon,
            min_instance_area=args.min_instance_area,
            min_component_area=args.min_component_area,
        )
        total.update(stats)

        image_dst = img_out / f"{frame.stem}.png"
        label_dst = lbl_out / f"{frame.stem}.txt"
        shutil.copy2(frame.image_path, image_dst)
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        rows.append(
            {
                "stem": frame.stem,
                "rig": frame.rig,
                "camera": "" if frame.camera is None else frame.camera,
                "plant": frame.plant,
                "sequence_id": frame.sequence_id,
                "day": frame.day,
                "time": frame.time,
                "frame_index": frame.frame_index,
                "hours": frame.hours,
                "width": width,
                "height": height,
                "n_leaves": len(lines),
                "leaf_ids": ";".join(map(str, leaf_ids)),
                "image": image_dst.relative_to(args.out).as_posix(),
                "label": label_dst.relative_to(args.out).as_posix(),
                "raw_label": frame.label_path.as_posix(),
            }
        )
        if i % 200 == 0:
            print(f"  converted {i}/{len(frames)}")

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(frames)} frames -> {args.out}")
    print(total.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
