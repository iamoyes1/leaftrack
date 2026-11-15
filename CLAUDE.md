# LeafTrack — project context

LeafTrack is a computer vision portfolio project for an Upwork profile. It must
look professional to clients who read the repo.

## Project goal

Leaf instance segmentation on top-down plant time-lapse images, stable leaf
IDs across frames (tracking), and growth analytics (leaf count, per-leaf area,
growth rate over time). Final deliverables: clean GitHub repo, live Gradio
demo on Hugging Face Spaces, README with demo GIF and results table.

## Constraints

- Training runs only on free Google Colab / Kaggle GPUs (T4). Use Ultralytics
  YOLO11n-seg or YOLO11s-seg. Notebooks are thin launchers that clone this
  repo and call `train.py`; all logic lives in the Python package.
- Checkpoints save to Google Drive; final weights go to Hugging Face Hub and
  are never committed to git.
- Demo runs on Hugging Face Spaces free CPU tier, so inference and the
  default tracker must run acceptably on CPU.
- Datasets (CVPPP leaf segmentation, KOMATSUNA time-lapse) live in `data/raw/`
  and are never committed.

## Decisions

- 2026-11-15: CVPPP is **not used**. Its terms forbid commercial use
  (including screenshots in advertising) and redistribution, which conflicts
  with a public portfolio, published weights and a live demo. Train and
  evaluate on KOMATSUNA only unless the owners grant written permission.
- Splits are by physical plant (`sequence_id`), never by frame or camera. The
  committed split lives in `configs/splits/komatsuna.json`.
- 2026-11-15: the user's local RTX 3050 Laptop GPU (6 GB) may also be used
  for training. The venv has the CUDA 13.0 build of torch. The Colab/Kaggle
  notebooks stay the documented, reproducible path.
- Multi-view crops contain leaves of neighbouring plants at the edges, which
  the labels (correctly) ignore. `LeafSegmenter` drops border-touching leaves
  that never come near the plant centre (`drop_neighbor_leaves`); on ground
  truth this removes 0 of 5,820 true leaves. Leaf metrics use a confidence
  threshold selected on the val split, never on test.
- Default training schedule is 80 epochs / patience 20 (YOLO11n peaked at
  epoch 73 of a 150-epoch run).
- KOMATSUNA label colors are stable per leaf over time, so they serve as
  ground-truth track IDs (kept per label line in `manifest.csv`).

## Standards

Type hints, docstrings, small focused modules, pytest tests, clear commit
messages. Work in phases and stop for review at the end of each phase.
