"""Resumable download and safe extraction of dataset archives."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

#: KOMATSUNA core files: name -> expected size in bytes (checked 2026-11-15).
KOMATSUNA_URL = "http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/"
KOMATSUNA_FILES: dict[str, int] = {
    "multi_readme.pdf": 201_682,
    "multi_plant.zip": 113_255_059,
    "multi_label.zip": 2_129_868,
    "rgbd_readme.pdf": 366_587,
    "rgbd_plant.zip": 9_339_573,
    "rgbd_label.zip": 299_553,
}


def download(url: str, dest: Path, expected_size: int | None = None, retries: int = 8) -> Path:
    """Download ``url`` to ``dest``, resuming partial files after failures.

    The KOMATSUNA server resets long transfers, so each retry continues from
    the bytes already on disk with an HTTP ``Range`` request.

    Raises:
        RuntimeError: if the file is still incomplete after all retries.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected_size is not None and have == expected_size:
            return dest
        if expected_size is not None and have > expected_size:
            dest.unlink()
            have = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = have > 0 and response.status == 206
                with open(dest, "ab" if append else "wb") as f:
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            print(f"  {dest.name}: attempt {attempt} interrupted ({exc}); resuming")
            time.sleep(min(2**attempt, 30))
            continue
        if expected_size is None:
            return dest
    size = dest.stat().st_size if dest.exists() else 0
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{dest.name}: got {size} of {expected_size} bytes")
    return dest


def safe_extract(archive: Path, target: Path) -> list[str]:
    """Extract a zip into ``target``, refusing entries that escape it.

    Returns:
        The archive's member names.

    Raises:
        ValueError: if any member has an absolute path or ``..`` component.
    """
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name in names:
            resolved = (target / name).resolve()
            if resolved != target and target not in resolved.parents:
                raise ValueError(f"Unsafe path in {archive.name}: {name}")
        zf.extractall(target)
    return names
