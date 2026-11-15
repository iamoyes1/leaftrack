"""Shared fixtures: a tiny synthetic dataset laid out exactly like KOMATSUNA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from leaftrack.datasets.komatsuna import LEAF_PALETTE


def make_label(h: int, w: int, leaf_ids: list[int], noise: bool = False) -> np.ndarray:
    """Draw one rectangle per leaf ID in its palette color on black."""
    label = np.zeros((h, w, 3), np.uint8)
    for k, leaf_id in enumerate(leaf_ids):
        y0 = 4 + 10 * k
        label[y0 : y0 + 8, 4 : w - 4] = LEAF_PALETTE[leaf_id]
    if noise:
        label[h - 2, w - 2] = (255, 255, 255)  # stray white pixel, as in real RGB-D labels
    return label


@pytest.fixture
def fake_komatsuna(tmp_path: Path) -> Path:
    """2 multi-view plants x 2 cameras x 2 frames + 2 RGB-D plants x 2 frames."""
    root = tmp_path / "komatsuna"
    for d in ["multi_plant", "multi_label", "rgbd_plant", "rgbd_label"]:
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(0)

    def write(img_dir: str, lbl_dir: str, suffix: str, h: int, w: int, ids: list[int]) -> None:
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(
            root / img_dir / f"rgb_{suffix}.png"
        )
        label = make_label(h, w, ids, noise=True)
        Image.fromarray(label).save(root / lbl_dir / f"label_{suffix}.png")

    for cam in (0, 1):
        for plant in (0, 1):
            for time in (0, 1):
                suffix = f"{cam:02d}_{plant:02d}_000_{time:02d}"
                write("multi_plant", "multi_label", suffix, 48, 48, [1, 2, 3])
    for plant in (0, 1):
        for time in (0, 1):
            write("rgbd_plant", "rgbd_label", f"{plant:02d}_000_{time:02d}", 40, 36, [2, 4])

    # Noise the reader must ignore: an unlabeled image and an unrelated file.
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "rgbd_plant" / "rgb_04_009_05.png")
    (root / "multi_plant" / "notes.txt").write_text("not an image")
    return root
