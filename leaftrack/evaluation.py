"""Test-set evaluation: COCO-style mAP, CVPPP leaf metrics and CPU latency."""

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from leaftrack.datasets.komatsuna import decode_label
from leaftrack.datasets.yolo import parse_yolo_line, polygon_to_mask
from leaftrack.metrics import ImageScore, score_image, summarize
from leaftrack.segment import LeafInstance, LeafSegmenter, drop_neighbor_leaves
from leaftrack.viz import draw_polygons


def load_split(data_dir: Path, split: str) -> list[dict[str, str]]:
    """Return manifest rows for the images listed in ``<data_dir>/<split>.txt``.

    Raises:
        FileNotFoundError: if the manifest or split list is missing.
        KeyError: if the split lists an image that is not in the manifest.
    """
    data_dir = Path(data_dir)
    with open(data_dir / "manifest.csv", newline="", encoding="utf-8") as f:
        by_image = {f"./{r['image']}": r for r in csv.DictReader(f)}
    listed = (data_dir / f"{split}.txt").read_text(encoding="utf-8").split()
    return [by_image[line] for line in listed]


def load_image(row: dict[str, str], data_dir: Path) -> np.ndarray:
    """Load a split image as an RGB uint8 array."""
    return np.asarray(Image.open(Path(data_dir) / row["image"]).convert("RGB"))


def load_gt_masks(row: dict[str, str], data_dir: Path) -> list[np.ndarray]:
    """Ground-truth leaf masks for one frame.

    Uses the original color label when it is available, so the evaluation is
    not capped by polygon conversion error, and otherwise falls back to the
    converted YOLO polygons.
    """
    raw = Path(row.get("raw_label", ""))
    if row.get("raw_label") and raw.exists():
        return list(decode_label(np.asarray(Image.open(raw).convert("RGB"))).values())
    w, h = int(row["width"]), int(row["height"])
    text = (Path(data_dir) / row["label"]).read_text(encoding="utf-8")
    return [
        polygon_to_mask(parse_yolo_line(line, w, h)[1], w, h).astype(bool)
        for line in text.splitlines()
        if line.strip()
    ]


Scored = list[tuple[dict[str, str], ImageScore]]


@dataclass(frozen=True)
class PostProcess:
    """Post-processing applied to raw model output before scoring.

    Attributes:
        conf: minimum detection confidence.
        drop_neighbors: remove leaves of adjacent plants at the image edge.
    """

    conf: float
    drop_neighbors: bool

    def apply(self, instances: list[LeafInstance]) -> list[LeafInstance]:
        kept = [i for i in instances if i.confidence >= self.conf]
        return drop_neighbor_leaves(kept) if self.drop_neighbors else kept


def sweep_postprocessing(
    segmenter: LeafSegmenter,
    rows: Sequence[dict[str, str]],
    data_dir: Path,
    settings: Sequence[PostProcess],
) -> dict[PostProcess, Scored]:
    """Score several post-processing settings with a single prediction pass.

    The segmenter must keep everything the settings might filter, i.e. its
    ``conf`` must not exceed the lowest setting and ``drop_neighbors`` must be
    off.

    Raises:
        ValueError: if the segmenter already filters more than a setting needs.
    """
    if segmenter.drop_neighbors or segmenter.conf > min(s.conf for s in settings):
        raise ValueError("Segmenter must use drop_neighbors=False and conf <= min(settings)")
    out: dict[PostProcess, Scored] = {s: [] for s in settings}
    for row in rows:
        image = load_image(row, data_dir)
        raw = segmenter.predict(image)
        gt = load_gt_masks(row, data_dir)
        for setting in settings:
            pred = [inst.mask for inst in setting.apply(raw)]
            out[setting].append((row, score_image(pred, gt, image.shape[:2])))
    return out


def select_setting(results: dict[PostProcess, Scored]) -> PostProcess:
    """Pick the setting with the best mean SBD (ties: lower mean |DiC|)."""

    def key(setting: PostProcess) -> tuple[float, float]:
        s = summarize(score for _, score in results[setting])
        return (s.sbd, -s.abs_dic)

    return max(results, key=key)


def evaluate_leaf_metrics(
    segmenter: LeafSegmenter, rows: Sequence[dict[str, str]], data_dir: Path
) -> Scored:
    """Run the segmenter (with its own post-processing) and score every row."""
    results = []
    for row in rows:
        image = load_image(row, data_dir)
        pred = [inst.mask for inst in segmenter.predict(image)]
        results.append((row, score_image(pred, load_gt_masks(row, data_dir), image.shape[:2])))
    return results


def summarize_by(
    results: Sequence[tuple[dict[str, str], ImageScore]], key: str = "rig"
) -> dict[str, dict[str, float]]:
    """Overall summary plus one summary per value of ``row[key]``."""
    out = {"all": summarize(s for _, s in results).to_dict()}
    for value in sorted({r[key] for r, _ in results}):
        out[value] = summarize(s for r, s in results if r[key] == value).to_dict()
    return out


def measure_latency(
    segmenter: LeafSegmenter, images: Sequence[np.ndarray], warmup: int = 3
) -> dict[str, float]:
    """Wall-clock inference latency (pre/post-processing included) in ms."""
    if not images:
        raise ValueError("Need at least one image to time")
    for image in list(images)[:warmup]:
        segmenter.predict(image)
    times = []
    for image in images:
        start = time.perf_counter()
        segmenter.predict(image)
        times.append((time.perf_counter() - start) * 1000)
    arr = np.asarray(times)
    return {
        "images": len(arr),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "mean_ms": float(arr.mean()),
    }


def ultralytics_val(
    weights: Path, data_yaml: Path, split: str, imgsz: int, device: str, out_dir: Path
) -> dict[str, float]:
    """COCO-style box and mask metrics from Ultralytics validation."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights), task="segment").val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        batch=8,
        project=str(Path(out_dir).resolve()),  # relative paths get nested under runs/
        name="ultralytics_val",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    return {
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
    }


def _outline(mask: np.ndarray) -> np.ndarray:
    """Largest external contour of a mask as an ``(N, 2)`` float array."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=len).reshape(-1, 2).astype(float)


def _letterbox(image: np.ndarray, cell: int) -> np.ndarray:
    """Centre an image on a white ``cell`` x ``cell`` square."""
    h, w = image.shape[:2]
    top, left = (cell - h) // 2, (cell - w) // 2
    out = np.full((cell, cell, 3), 255, np.uint8)
    out[top : top + h, left : left + w] = image
    return out


def example_panel(
    image: np.ndarray, pred: Sequence[LeafInstance], gt: Sequence[np.ndarray], cell: int = 320
) -> np.ndarray:
    """``image | prediction | ground truth`` strip of three square ``cell`` tiles."""
    scale = cell / max(image.shape[:2])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    big = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    pred_polys = [_outline(cv2.resize(p.mask.astype(np.uint8), size) > 0) for p in pred]
    gt_polys = [_outline(cv2.resize(g.astype(np.uint8), size) > 0) for g in gt]
    tiles = [big, draw_polygons(big, pred_polys), draw_polygons(big, gt_polys)]
    gap = np.full((cell, 4, 3), 255, np.uint8)
    boxed = [_letterbox(t, cell) for t in tiles]
    return np.hstack([boxed[0], gap, boxed[1], gap, boxed[2]])


def example_grid(panels: Sequence[np.ndarray], columns: int = 2) -> np.ndarray:
    """Tile equally sized panels into a grid with white gutters and a header row."""
    padded = list(panels)
    while len(padded) % columns:
        padded.append(np.full_like(padded[0], 255))
    gutter_v = np.full((padded[0].shape[0], 12, 3), 255, np.uint8)
    rows = [
        np.hstack([x for p in padded[i : i + columns] for x in (p, gutter_v)][:-1])
        for i in range(0, len(padded), columns)
    ]
    gutter_h = np.full((12, rows[0].shape[1], 3), 255, np.uint8)
    body = np.vstack([x for r in rows for x in (r, gutter_h)][:-1])
    header = np.full((30, body.shape[1], 3), 255, np.uint8)
    cell = padded[0].shape[0]
    for c in range(columns):
        x0 = c * (padded[0].shape[1] + 12)
        for k, label in enumerate(["image", "prediction", "ground truth"]):
            org = (x0 + k * (cell + 4) + 8, 21)
            cv2.putText(
                header, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA
            )
    return np.vstack([header, body])


def pick_examples(rows: Sequence[dict[str, str]], n: int) -> list[dict[str, str]]:
    """Spread ``n`` examples over rigs and growth stages, deterministically."""
    by_rig: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_rig.setdefault(r["rig"], []).append(r)
    rigs = sorted(by_rig)
    picks = []
    for k in range(n):
        pool = sorted(by_rig[rigs[k % len(rigs)]], key=lambda r: int(r["frame_index"]))
        stage = (k // len(rigs)) / max(1, (n + len(rigs) - 1) // len(rigs) - 1)
        picks.append(pool[min(len(pool) - 1, round(stage * (len(pool) - 1)))])
    return picks


def results_markdown(name: str, report: dict[str, Any]) -> str:
    """Render one evaluation report as Markdown tables for the README."""
    leaf = report["leaf_metrics"]
    lines = [f"### {name}", ""]
    if report.get("map"):
        m = report["map"]
        lines += [
            "| Mask mAP50 | Mask mAP50-95 | Box mAP50 | Box mAP50-95 |",
            "|---|---|---|---|",
            f"| {m['mask_map50']:.3f} | {m['mask_map50_95']:.3f} "
            f"| {m['box_map50']:.3f} | {m['box_map50_95']:.3f} |",
            "",
        ]
    if report.get("postprocess"):
        pp = report["postprocess"]
        neighbors = "on" if pp["drop_neighbors"] else "off"
        lines += [
            f"Leaf metrics at confidence {pp['conf']:.2f} (selected on {pp['selected_on']}), "
            f"neighbour-leaf filter {neighbors}:",
            "",
        ]
    lines += [
        "| Subset | Images | SBD ↑ | FBD ↑ | \\|DiC\\| ↓ | DiC | Exact count |",
        "|---|---|---|---|---|---|---|",
    ]
    for subset, s in leaf.items():
        lines.append(
            f"| {subset} | {s['images']} | {s['sbd']:.3f} ± {s['sbd_std']:.3f} "
            f"| {s['fbd']:.3f} | {s['abs_dic']:.2f} | {s['dic']:+.2f} "
            f"| {s['count_accuracy']:.0%} |"
        )
    if report.get("ablation"):
        lines += [
            "",
            "Effect of post-processing (test, all images):",
            "",
            "| Setting | SBD ↑ | \\|DiC\\| ↓ | Exact count |",
            "|---|---|---|---|",
        ]
        for a in report["ablation"]:
            lines.append(
                f"| {a['label']} | {a['sbd']:.3f} | {a['abs_dic']:.2f} "
                f"| {a['count_accuracy']:.0%} |"
            )
    if report.get("latency"):
        lat = report["latency"]
        lines += [
            "",
            f"CPU latency ({lat['threads']} threads, imgsz {lat['imgsz']}): "
            f"median {lat['median_ms']:.0f} ms, p90 {lat['p90_ms']:.0f} ms per image.",
        ]
    return "\n".join(lines) + "\n"
