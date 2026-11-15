"""End-to-end test: convert -> split -> visualize on a synthetic KOMATSUNA tree."""

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "data" / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_convert_split_visualize(fake_komatsuna: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    split_file = tmp_path / "split.json"

    convert = load_script("convert_komatsuna")
    assert convert.main(["--raw", str(fake_komatsuna), "--out", str(out)]) == 0
    with open(out / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    for r in rows:
        label_lines = (out / r["label"]).read_text().strip().splitlines()
        assert len(label_lines) == int(r["n_leaves"]) == len(r["leaf_ids"].split(";"))
        assert (out / r["image"]).exists()

    split = load_script("split_dataset")
    args = ["--data", str(out), "--split-file", str(split_file), "--ratios", "0.34", "0.33", "0.33"]
    assert split.main(args) == 0
    saved = json.loads(split_file.read_text())["groups"]

    seq_of = {f"./{r['image']}": r["sequence_id"] for r in rows}
    seqs_per_split = {
        name: {seq_of[line] for line in (out / f"{name}.txt").read_text().split()}
        for name in ("train", "val", "test")
    }
    all_listed = [s for seqs in seqs_per_split.values() for s in seqs]
    assert len(all_listed) == len(set(all_listed)), "a plant appears in more than one split"
    assert {k: sorted(v) for k, v in seqs_per_split.items() if v} == {
        k: sorted(v) for k, v in saved.items() if v
    }

    cfg = yaml.safe_load((out / "dataset.yaml").read_text())
    assert cfg["names"] == {0: "leaf"} and Path(cfg["path"]).is_absolute()

    # Re-running reuses the saved split instead of drawing a new one.
    assert split.main(args[:4] + ["--seed", "999"]) == 0
    assert json.loads(split_file.read_text())["groups"] == saved

    check = tmp_path / "check"
    viz_args = ["--data", str(out), "--out", str(check), "--n", "3", "--panel-height", "96"]
    assert load_script("visualize_labels").main(viz_args) == 0
    assert len(list(check.glob("*.png"))) == 3
