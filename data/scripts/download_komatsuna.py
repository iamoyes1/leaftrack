"""Download and extract the KOMATSUNA core files (about 125 MB).

Archives are kept in ``--archives`` (point it at Google Drive on Colab so the
slow download happens once) and extracted into ``--out``. Complete archives
and already-extracted folders are skipped, so re-running is cheap.

Usage::

    python data/scripts/download_komatsuna.py \
        --archives /content/drive/MyDrive/leaftrack/archives --out data/raw/komatsuna
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from leaftrack.datasets.download import KOMATSUNA_FILES, KOMATSUNA_URL, download, safe_extract

EXPECTED_IMAGES = {"multi_plant": 900, "multi_label": 900, "rgbd_plant": 300, "rgbd_label": 300}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--archives", type=Path, default=Path("data/raw/komatsuna/_archives"))
    p.add_argument("--out", type=Path, default=Path("data/raw/komatsuna"))
    args = p.parse_args(argv)

    for name, size in KOMATSUNA_FILES.items():
        path = download(KOMATSUNA_URL + name, args.archives / name, expected_size=size)
        print(f"  ok  {path.name} ({size / 1e6:.1f} MB)")
        if name.endswith(".pdf"):
            args.out.mkdir(parents=True, exist_ok=True)
            if path.resolve() != (args.out / name).resolve():
                shutil.copy2(path, args.out / name)
            continue
        folder = args.out / name.removesuffix(".zip")
        if len(list(folder.glob("*.png"))) == EXPECTED_IMAGES[folder.name]:
            continue
        safe_extract(path, args.out)
        print(f"  extracted {folder} ({len(list(folder.glob('*.png')))} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
