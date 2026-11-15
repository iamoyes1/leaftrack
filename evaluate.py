"""Evaluate trained weights on the held-out test split.

Reports, for the README results table:

* Ultralytics COCO-style mask and box mAP (threshold-free; ``--skip-map``
  to omit),
* CVPPP leaf metrics (SBD, FBD, DiC, |DiC|) overall and per capture rig,
  scored against the original color labels. These depend on a confidence
  threshold, which is chosen on the **validation** plants (best mean SBD) so
  the test plants stay unseen. Pass ``--conf`` to fix it instead,
* a small post-processing ablation on test (default threshold vs selected
  threshold, with and without the neighbour-leaf filter),
* CPU latency with a fixed thread count, to mirror the free Hugging Face
  Spaces CPU tier (2 vCPU).

Example::

    python evaluate.py --weights runs/segment/yolo11n/weights/best.pt --device 0

Writes ``metrics.json``, ``results.md`` and ``per_image.csv`` to ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from leaftrack.evaluation import (
    PostProcess,
    example_grid,
    example_panel,
    load_gt_masks,
    load_image,
    load_split,
    measure_latency,
    pick_examples,
    results_markdown,
    select_setting,
    summarize_by,
    sweep_postprocessing,
    ultralytics_val,
)
from leaftrack.metrics import summarize
from leaftrack.segment import LeafSegmenter

DEFAULT_CONF = 0.25  # Ultralytics' default, shown in the ablation as the baseline
CONF_GRID = [round(0.25 + 0.05 * i, 2) for i in range(13)]  # 0.25 ... 0.85


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/processed/komatsuna"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="cpu", help="'cpu' or a GPU index such as '0'")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, help="fixed threshold (default: select on val)")
    p.add_argument("--no-neighbor-filter", action="store_true", help="keep edge leaves")
    p.add_argument("--name", help="report title (default: the run folder name)")
    p.add_argument("--out", type=Path, help="output folder (default: outputs/eval/<name>)")
    p.add_argument("--skip-map", action="store_true", help="skip Ultralytics mAP validation")
    p.add_argument("--examples", type=int, default=6, help="example panels to render (0: none)")
    p.add_argument("--latency-images", type=int, default=50, help="0 disables CPU timing")
    p.add_argument("--cpu-threads", type=int, default=2, help="torch threads for CPU timing")
    return p.parse_args(argv)


def default_name(weights: Path) -> str:
    """``runs/segment/yolo11n/weights/best.pt`` -> ``yolo11n``."""
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    name = args.name or default_name(args.weights)
    out = args.out or Path("outputs/eval") / name
    out.mkdir(parents=True, exist_ok=True)
    rows = load_split(args.data, args.split)
    drop = not args.no_neighbor_filter
    print(f"Evaluating {args.weights} on {len(rows)} {args.split} images")

    report: dict = {"weights": str(args.weights), "split": args.split, "imgsz": args.imgsz}
    if not args.skip_map:
        report["map"] = ultralytics_val(
            args.weights, args.data / "dataset.yaml", args.split, args.imgsz, args.device, out
        )
        print(f"  mask mAP50-95 = {report['map']['mask_map50_95']:.3f}")

    # One low-threshold prediction pass per split; thresholds are applied after.
    raw = LeafSegmenter(
        args.weights, device=args.device, conf=min(CONF_GRID), imgsz=args.imgsz,
        drop_neighbors=False,
    )  # fmt: skip
    if args.conf is None:
        val_rows = load_split(args.data, "val")
        grid = [PostProcess(c, drop) for c in CONF_GRID]
        val = sweep_postprocessing(raw, val_rows, args.data, grid)
        chosen = select_setting(val)
        selected_on = f"val ({len(val_rows)} images)"
    else:
        chosen, selected_on = PostProcess(args.conf, drop), "fixed by --conf"
    report["postprocess"] = {
        "conf": chosen.conf, "drop_neighbors": chosen.drop_neighbors, "selected_on": selected_on,
    }  # fmt: skip
    print(f"  confidence threshold = {chosen.conf} ({selected_on})")

    variants = {
        f"conf {DEFAULT_CONF} (Ultralytics default)": PostProcess(DEFAULT_CONF, False),
        f"conf {chosen.conf} (selected on val)": PostProcess(chosen.conf, False),
        f"conf {chosen.conf} + neighbour-leaf filter": PostProcess(chosen.conf, True),
    }
    settings = list(dict.fromkeys([*variants.values(), chosen]))
    test = sweep_postprocessing(raw, rows, args.data, settings)
    results = test[chosen]
    report["leaf_metrics"] = summarize_by(results)
    report["ablation"] = [
        {"label": label, **summarize(s for _, s in test[setting]).to_dict()}
        for label, setting in variants.items()
    ]
    print(f"  SBD = {report['leaf_metrics']['all']['sbd']:.3f}")

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "rig", "sbd", "fbd", "dic", "n_pred", "n_gt"])
        for row, s in results:
            writer.writerow(
                [row["stem"], row["rig"], f"{s.sbd:.4f}", f"{s.fbd:.4f}", s.dic, s.n_pred, s.n_gt]
            )

    if args.examples > 0:
        final = LeafSegmenter(
            args.weights, device=args.device, conf=chosen.conf, imgsz=args.imgsz,
            drop_neighbors=chosen.drop_neighbors,
        )  # fmt: skip
        panels = []
        for row in pick_examples(rows, args.examples):
            image = load_image(row, args.data)
            panels.append(example_panel(image, final.predict(image), load_gt_masks(row, args.data)))
        Image.fromarray(example_grid(panels)).save(out / "examples.png")
        print(f"  wrote {out / 'examples.png'}")

    if args.latency_images > 0:
        import torch

        torch.set_num_threads(args.cpu_threads)
        cpu = LeafSegmenter(
            args.weights, device="cpu", conf=chosen.conf, imgsz=args.imgsz,
            drop_neighbors=chosen.drop_neighbors,
        )  # fmt: skip
        step = max(1, len(rows) // args.latency_images)
        images = [load_image(r, args.data) for r in rows[::step][: args.latency_images]]
        report["latency"] = measure_latency(cpu, images) | {
            "threads": args.cpu_threads,
            "imgsz": args.imgsz,
        }
        print(f"  CPU median latency = {report['latency']['median_ms']:.0f} ms")

    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "results.md").write_text(results_markdown(name, report), encoding="utf-8")
    print(f"Wrote {out / 'metrics.json'} and {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
