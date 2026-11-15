"""Train a YOLO11-seg leaf segmentation model.

This is the single entry point used by the Colab / Kaggle launcher notebooks.
Point ``--project`` at Google Drive so checkpoints survive notebook resets,
and add ``--resume`` after a disconnect to continue from ``last.pt``.

Examples::

    # Colab (checkpoints on Drive)
    python train.py --config configs/train_yolo11n.yaml \
        --project /content/drive/MyDrive/leaftrack/runs --name yolo11n

    # Continue after the session died
    python train.py --config configs/train_yolo11n.yaml \
        --project /content/drive/MyDrive/leaftrack/runs --name yolo11n --resume

    # Quick CPU smoke test
    python train.py --config configs/train_yolo11n.yaml --name smoke \
        --set epochs=1 imgsz=320 batch=4 fraction=0.05 device=cpu cache=false
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leaftrack.training import load_config, parse_overrides, plan_training, run_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="training YAML in configs/")
    p.add_argument("--data", type=Path, help="dataset.yaml (default: the config's 'data')")
    p.add_argument("--project", type=Path, default=Path("runs/segment"), help="run root folder")
    p.add_argument("--name", help="run name (default: config file name)")
    p.add_argument("--resume", action="store_true", help="continue from <run>/weights/last.pt")
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override Ultralytics train settings, e.g. --set epochs=1 device=cpu",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    name = args.name or args.config.stem.removeprefix("train_")
    plan = plan_training(
        cfg,
        project=args.project,
        name=name,
        data=args.data,
        resume=args.resume,
        overrides=parse_overrides(args.set),
    )
    action = "Resuming" if plan.resume else "Starting"
    print(f"{action} run {plan.run_dir} from {plan.model}")
    best = run_training(plan)
    print(f"Best weights: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
