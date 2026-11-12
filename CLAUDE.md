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

## Standards

Type hints, docstrings, small focused modules, pytest tests, clear commit
messages. Work in phases and stop for review at the end of each phase.
