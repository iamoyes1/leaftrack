"""Publish trained weights to the Hugging Face Hub (never to git).

Usage::

    export HF_TOKEN=...   # a write token; or run `huggingface-cli login`
    python -m leaftrack.hub --weights runs/segment/yolo11n/weights/best.pt \
        --repo <user>/leaftrack-yolo11n-seg --metrics outputs/eval/yolo11n/metrics.json

Uploads ``best.pt``, the evaluation ``metrics.json`` (if given) and a model
card generated from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CARD_TEMPLATE = """---
license: mit
library_name: ultralytics
pipeline_tag: image-segmentation
tags:
  - yolo11
  - instance-segmentation
  - plant-phenotyping
  - leaf-segmentation
---

# LeafTrack: {title}

YOLO11-seg model that segments individual leaves in top-down plant images.
Trained with [LeafTrack](https://github.com/iamoyes1/leaftrack) on the
[KOMATSUNA](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) dataset
(Uchiyama et al., 2017), split by plant so test plants were never seen in training.

## Usage

```python
from leaftrack.segment import LeafSegmenter

segmenter = LeafSegmenter.from_hub("{repo_id}")
leaves = segmenter.predict(rgb_image)  # list of LeafInstance (mask, box, confidence)
```

## Test-set results

{results}

## Training data and license

The weights are released under MIT. The KOMATSUNA dataset is credited to
Uchiyama et al., "An Easy-to-Setup 3D Phenotyping Platform for KOMATSUNA
Dataset", ICCV Workshops 2017.
"""


def model_card(repo_id: str, title: str, metrics: dict[str, Any] | None) -> str:
    """Render the model card README, embedding evaluation results if present."""
    if metrics:
        from leaftrack.evaluation import results_markdown

        results = results_markdown("Test split", metrics).split("\n", 2)[2]
    else:
        results = "Not yet evaluated."
    return CARD_TEMPLATE.format(title=title, repo_id=repo_id, results=results)


def push_model(
    weights: Path,
    repo_id: str,
    metrics_path: Path | None = None,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Create (if needed) a model repo and upload weights, metrics and card.

    ``token`` defaults to the ``HF_TOKEN`` environment variable or the cached
    ``huggingface-cli login`` credentials.

    Returns:
        The URL of the model repo.
    """
    from huggingface_hub import HfApi

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    metrics = json.loads(Path(metrics_path).read_text()) if metrics_path else None

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(weights), path_in_repo="best.pt", repo_id=repo_id)
    if metrics_path:
        api.upload_file(
            path_or_fileobj=str(metrics_path), path_in_repo="metrics.json", repo_id=repo_id
        )
    card = model_card(repo_id, repo_id.split("/")[-1], metrics)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def use_system_certificates() -> bool:
    """Verify HTTPS against the OS certificate store when ``truststore`` is installed.

    Needed on networks that re-sign TLS traffic with a certificate the OS
    trusts but Python's bundled CA list does not. Verification stays on.
    """
    try:
        import truststore
    except ImportError:
        return False
    truststore.inject_into_ssl()
    return True


def main(argv: list[str] | None = None) -> int:
    use_system_certificates()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--repo", required=True, help="<user>/<model-name>")
    p.add_argument("--metrics", type=Path, help="metrics.json written by evaluate.py")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)
    print(push_model(args.weights, args.repo, args.metrics, args.private))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
