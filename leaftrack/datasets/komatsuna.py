"""Reader for the KOMATSUNA time-lapse leaf dataset (Uchiyama et al., 2017).

Facts below were checked against the downloaded files, not only the readme:

* Two capture rigs with *different* physical plants (5 plants each):

  - ``multi``: three fixed RGB cameras (00 right, 01 center, 02 left) looking
    down on the same tray. Files: ``multi_plant/rgb_CC_PP_DDD_TT.png`` and
    ``multi_label/label_CC_PP_DDD_TT.png``. Crops are 480x480 (plant 04:
    544x480).
  - ``rgbd``: one top-down Intel RealSense SR300. Files:
    ``rgbd_plant/rgb_PP_DDD_TT.png`` and ``rgbd_label/label_PP_DDD_TT.png``.
    Crops are 166x190 (plants 00-02) or 204x185 (plants 03-04).

* Time: day ``DDD`` in 000-009, time slot ``TT`` in 00-05, one capture every
  4 hours, so each plant has 60 frames. Every frame has a label.

* Labels are RGB PNGs on a black background. Each leaf is painted with a fixed
  color assigned by leaf age (oldest first), and a leaf keeps its color for the
  whole sequence. The color index is therefore a ground-truth *track ID*.
  A few RGB-D labels contain isolated white pixels (1-7 px, 55 px in total);
  they are not a palette color and are treated as background.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Leaf colors in label images, keyed by leaf ID (1 = oldest leaf).
LEAF_PALETTE: dict[int, tuple[int, int, int]] = {
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (0, 255, 255),
    4: (255, 0, 0),
    5: (255, 0, 255),
    6: (255, 255, 0),
    7: (128, 128, 128),
    8: (0, 0, 128),
}

FRAMES_PER_DAY = 6
HOURS_PER_FRAME = 4

RIGS: dict[str, tuple[str, str, re.Pattern[str]]] = {
    "multi": (
        "multi_plant",
        "multi_label",
        re.compile(r"^rgb_(?P<camera>\d{2})_(?P<plant>\d{2})_(?P<day>\d{3})_(?P<time>\d{2})\.png$"),
    ),
    "rgbd": (
        "rgbd_plant",
        "rgbd_label",
        re.compile(r"^rgb_(?P<plant>\d{2})_(?P<day>\d{3})_(?P<time>\d{2})\.png$"),
    ),
}


@dataclass(frozen=True)
class KomatsunaFrame:
    """One labelled image of one plant at one time point.

    Attributes:
        rig: ``"multi"`` or ``"rgbd"``.
        camera: camera ID for the multi-view rig, ``None`` for RGB-D.
        plant: plant ID within the rig (0-4).
        day: day index (0-9).
        time: time slot within the day (0-5).
        image_path: path to the RGB plant crop.
        label_path: path to the RGB label image.
    """

    rig: str
    camera: int | None
    plant: int
    day: int
    time: int
    image_path: Path
    label_path: Path

    @property
    def sequence_id(self) -> str:
        """Physical-plant identifier. All cameras of one plant share it."""
        return f"{self.rig}_p{self.plant:02d}"

    @property
    def frame_index(self) -> int:
        """Position in the plant's time-lapse (0-59)."""
        return self.day * FRAMES_PER_DAY + self.time

    @property
    def hours(self) -> int:
        """Hours since the first capture of the sequence."""
        return self.frame_index * HOURS_PER_FRAME

    @property
    def stem(self) -> str:
        """Unique, self-describing output file stem."""
        cam = f"_c{self.camera:02d}" if self.camera is not None else ""
        return f"komatsuna_{self.rig}{cam}_p{self.plant:02d}_d{self.day:03d}_t{self.time:02d}"


def parse_image_name(name: str, rig: str) -> dict[str, int | None] | None:
    """Parse a KOMATSUNA plant-image file name.

    Returns:
        ``{"camera", "plant", "day", "time"}`` (``camera`` is ``None`` for the
        RGB-D rig), or ``None`` if the name does not match the rig's pattern.

    Raises:
        KeyError: if ``rig`` is unknown.
    """
    m = RIGS[rig][2].match(name)
    if m is None:
        return None
    groups = m.groupdict()
    return {
        "camera": int(groups["camera"]) if groups.get("camera") is not None else None,
        "plant": int(groups["plant"]),
        "day": int(groups["day"]),
        "time": int(groups["time"]),
    }


def discover_frames(root: Path) -> Iterator[KomatsunaFrame]:
    """Yield every frame under ``root`` that has both an image and a label.

    ``root`` is the extracted dataset folder containing ``multi_plant/``,
    ``multi_label/``, ``rgbd_plant/`` and ``rgbd_label/`` (rigs whose folders
    are missing are skipped). Frames are yielded in sorted file-name order.
    """
    root = Path(root)
    for rig, (img_dir, lbl_dir, _) in RIGS.items():
        img_root, lbl_root = root / img_dir, root / lbl_dir
        if not img_root.is_dir() or not lbl_root.is_dir():
            continue
        for image_path in sorted(img_root.glob("rgb_*.png")):
            meta = parse_image_name(image_path.name, rig)
            if meta is None:
                continue
            label_path = lbl_root / ("label_" + image_path.name[len("rgb_") :])
            if not label_path.exists():
                continue
            yield KomatsunaFrame(
                rig=rig,
                image_path=image_path,
                label_path=label_path,
                **meta,  # type: ignore[arg-type]
            )


def decode_label(label_rgb: np.ndarray) -> dict[int, np.ndarray]:
    """Split an RGB label image into one boolean mask per leaf.

    Args:
        label_rgb: ``(H, W, 3)`` uint8 array in RGB channel order.

    Returns:
        ``{leaf_id: mask}`` for every palette color present, sorted by leaf ID.
        Pixels whose color is not in :data:`LEAF_PALETTE` are ignored.
    """
    arr = np.asarray(label_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB array, got shape {arr.shape}.")
    arr = arr.astype(np.int32)
    code = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
    present = set(np.unique(code).tolist())
    masks: dict[int, np.ndarray] = {}
    for leaf_id, (r, g, b) in LEAF_PALETTE.items():
        value = (r << 16) | (g << 8) | b
        if value in present:
            masks[leaf_id] = code == value
    return masks
