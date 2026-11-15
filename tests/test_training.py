"""Tests for training planning (no GPU or Ultralytics run needed)."""

from pathlib import Path

import pytest
import yaml

from leaftrack.training import load_config, parse_overrides, plan_training

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["train_yolo11n.yaml", "train_yolo11s.yaml"])
def test_shipped_configs_are_valid(name: str) -> None:
    cfg = load_config(ROOT / "configs" / name)
    assert cfg["model"].endswith("-seg.pt")
    assert cfg["train"]["epochs"] > 0


def test_config_rejects_reserved_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"model": "m.pt", "train": {"project": "x"}}))
    with pytest.raises(ValueError, match="project"):
        load_config(path)


def test_config_requires_model(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"train": {}}))
    with pytest.raises(ValueError, match="model"):
        load_config(path)


def test_parse_overrides_types_values() -> None:
    assert parse_overrides(["epochs=1", "fraction=0.1", "device=cpu", "cache=false"]) == {
        "epochs": 1,
        "fraction": 0.1,
        "device": "cpu",
        "cache": False,
    }
    with pytest.raises(ValueError):
        parse_overrides(["epochs"])


def test_fresh_plan_merges_config_and_overrides(tmp_path: Path) -> None:
    cfg = {"model": "yolo11n-seg.pt", "data": "d.yaml", "train": {"epochs": 150, "batch": 16}}
    plan = plan_training(cfg, project=tmp_path, name="run", overrides={"epochs": 1})
    assert not plan.resume and plan.model == "yolo11n-seg.pt"
    assert plan.kwargs["epochs"] == 1 and plan.kwargs["batch"] == 16
    assert plan.kwargs["data"] == "d.yaml"
    assert plan.kwargs["project"] == str(tmp_path) and plan.kwargs["name"] == "run"
    assert plan.run_dir == tmp_path.resolve() / "run"
    assert cfg["train"]["epochs"] == 150, "config must not be mutated"


def test_relative_project_is_made_absolute(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    plan = plan_training({"model": "m.pt"}, project=Path("runs/segment"), name="run")
    assert Path(plan.kwargs["project"]).is_absolute()
    assert plan.run_dir == (tmp_path / "runs" / "segment" / "run").resolve()


def test_resume_uses_last_checkpoint_when_present(tmp_path: Path) -> None:
    cfg = {"model": "yolo11n-seg.pt", "train": {}}
    no_ckpt = plan_training(cfg, project=tmp_path, name="run", resume=True)
    assert not no_ckpt.resume, "nothing to resume yet: start fresh"

    last = tmp_path / "run" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"x")
    plan = plan_training(cfg, project=tmp_path, name="run", resume=True)
    assert plan.resume and plan.model == str(last) and plan.kwargs == {"resume": True}


def test_overrides_cannot_hijack_cli_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_training({"model": "m.pt"}, tmp_path, "run", overrides={"data": "other.yaml"})
