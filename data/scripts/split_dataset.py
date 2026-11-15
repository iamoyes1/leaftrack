"""Create a leakage-free train/val/test split and the Ultralytics dataset YAML.

Splits are made over physical plants (``sequence_id``), never over frames:
all 60 time points of a plant, from every camera that saw it, go to the same
split. Plants are stratified by capture rig so each split contains both the
multi-view and the RGB-D setup.

The plant-level assignment is saved to ``configs/splits/komatsuna.json``
(committed, so every machine trains on the same split). If that file exists
it is reused; pass ``--resplit`` to draw a new one.

Usage::

    python data/scripts/split_dataset.py --data data/processed/komatsuna

Writes ``train.txt``, ``val.txt``, ``test.txt`` (image paths relative to the
data folder) and ``dataset.yaml`` into ``--data``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from leaftrack.datasets.split import SPLITS, assign_items, split_groups


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed/komatsuna"))
    p.add_argument("--split-file", type=Path, default=Path("configs/splits/komatsuna.json"))
    p.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resplit", action="store_true", help="ignore an existing split file")
    return p.parse_args(argv)


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Read the converter's ``manifest.csv``."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_manifest(args.data / "manifest.csv")
    group_of = {r["stem"]: r["sequence_id"] for r in rows}
    rig_of = {r["sequence_id"]: r["rig"] for r in rows}

    if args.split_file.exists() and not args.resplit:
        group_split = json.loads(args.split_file.read_text(encoding="utf-8"))["groups"]
        print(f"Reusing plant-level split from {args.split_file}")
    else:
        group_split = split_groups(
            rig_of.keys(), ratios=tuple(args.ratios), seed=args.seed, strata=rig_of
        )
        args.split_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "komatsuna",
            "unit": "sequence_id (one physical plant, all cameras and time points)",
            "seed": args.seed,
            "ratios": list(args.ratios),
            "groups": group_split,
        }
        args.split_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote new plant-level split to {args.split_file}")

    missing = set(group_of.values()) - {g for gs in group_split.values() for g in gs}
    if missing:
        print(f"Split file does not cover sequences: {sorted(missing)}", file=sys.stderr)
        return 1

    item_split = assign_items(group_of.keys(), group_of, group_split)
    image_of = {r["stem"]: r["image"] for r in rows}
    leaves_of = {r["stem"]: int(r["n_leaves"]) for r in rows}
    for name in SPLITS:
        lines = [f"./{image_of[s]}" for s in item_split[name]]
        (args.data / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rigs = Counter(rig_of[group_of[s]] for s in item_split[name])
        print(
            f"  {name:5s}: plants={group_split[name]} images={len(lines)} "
            f"leaves={sum(leaves_of[s] for s in item_split[name])} rigs={dict(rigs)}"
        )

    dataset_yaml = {
        "path": str(args.data.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "leaf"},
    }
    with open(args.data / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)
    print(f"Wrote {args.data / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
