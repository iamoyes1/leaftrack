"""Training orchestration around Ultralytics YOLO11-seg.

Everything except :func:`run_training` is plain Python so it can be unit
tested without a GPU or Ultralytics installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Keys the CLI controls; they may not be set in a config's ``train`` section.
RESERVED_KEYS = frozenset({"data", "project", "name", "resume", "model"})


@dataclass(frozen=True)
class TrainPlan:
    """Everything needed to launch (or resume) one training run.

    Attributes:
        model: weights to start from (e.g. ``yolo11n-seg.pt``), or the run's
            ``last.pt`` when resuming.
        resume: whether this continues an interrupted run.
        kwargs: keyword arguments for ``YOLO.train()``.
        run_dir: directory Ultralytics writes the run into.
    """

    model: str
    resume: bool
    kwargs: dict[str, Any]
    run_dir: Path


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a training config YAML.

    Raises:
        ValueError: if ``model`` is missing or ``train`` uses a reserved key.
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "model" not in cfg:
        raise ValueError(f"{path}: config must define 'model'")
    train = cfg.setdefault("train", {}) or {}
    clash = RESERVED_KEYS & set(train)
    if clash:
        raise ValueError(f"{path}: set {sorted(clash)} via the CLI, not under 'train'")
    cfg["train"] = train
    return cfg


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` CLI overrides, YAML-typing each value.

    Example: ``["epochs=1", "fraction=0.1", "device=cpu"]`` becomes
    ``{"epochs": 1, "fraction": 0.1, "device": "cpu"}``.
    """
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"Override must look like key=value, got {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def plan_training(
    cfg: Mapping[str, Any],
    project: Path,
    name: str,
    data: Path | None = None,
    resume: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> TrainPlan:
    """Turn a config plus CLI arguments into a concrete :class:`TrainPlan`.

    When ``resume`` is set and ``<project>/<name>/weights/last.pt`` exists, the
    plan continues that run (Ultralytics restores epochs, optimizer and
    hyperparameters from the checkpoint). Without a checkpoint it starts fresh.

    Raises:
        ValueError: if an override uses a reserved key.
    """
    overrides = dict(overrides or {})
    clash = RESERVED_KEYS & set(overrides)
    if clash:
        raise ValueError(f"Use the dedicated CLI flags for {sorted(clash)}")

    # Ultralytics nests a *relative* project under its own runs/ folder, which
    # would make the run land somewhere other than where --resume looks.
    project = Path(project).resolve()
    run_dir = project / name
    last = run_dir / "weights" / "last.pt"
    if resume and last.exists():
        return TrainPlan(model=str(last), resume=True, kwargs={"resume": True}, run_dir=run_dir)

    kwargs: dict[str, Any] = dict(cfg.get("train", {}))
    kwargs.update(overrides)
    kwargs.update(
        data=str(data or cfg.get("data")),
        project=str(project),
        name=name,
        exist_ok=True,  # re-running a fresh plan into the same folder is fine
    )
    return TrainPlan(model=str(cfg["model"]), resume=False, kwargs=kwargs, run_dir=run_dir)


def run_training(plan: TrainPlan) -> Path:
    """Execute a plan with Ultralytics and return the path to ``best.pt``.

    Raises:
        FileNotFoundError: if training finished without producing ``best.pt``.
    """
    from ultralytics import YOLO  # imported lazily: heavy, and only needed here

    model = YOLO(plan.model)
    model.train(**plan.kwargs)
    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir) if trainer is not None else plan.run_dir
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but {best} was not written")
    return best
