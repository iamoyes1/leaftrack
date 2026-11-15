# LeafTrack

**Leaf instance segmentation, tracking, and growth analytics for top-down plant time-lapse imagery.**

> Work in progress. The demo GIF, live Hugging Face Space, and results table will be added as the project develops.

## What it does

1. **Segment** every leaf in a top-down plant image with a YOLO11-seg model.
2. **Track** each leaf across a time-lapse sequence so it keeps a stable ID.
3. **Analyze** growth: leaf count, per-leaf area, and growth rate over time.

## Project structure

```
leaftrack/        Core package: segment, track, metrics, analytics
data/scripts/     Dataset conversion (to YOLO-seg), splitting, visual checks
configs/          Training and dataset configs
train.py          Training entry point (used by Colab/Kaggle notebooks)
evaluate.py       Test-set evaluation and results table
api/              FastAPI service
app.py            Gradio demo (Hugging Face Spaces)
notebooks/        Thin GPU launcher notebooks
tests/            pytest suite
```

## Dataset

[KOMATSUNA](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) (Uchiyama et al., 2017): 1,200 labelled top-down frames of 10 plants over 10 days, with leaf identities that stay consistent over time. That makes it a fit for both segmentation and tracking evaluation.

The data is converted to YOLO-seg polygons and split **by plant, not by frame**, so no plant seen in training appears in validation or test. See [data/README.md](data/README.md) for the verified data format, conversion quality and split. Datasets and model weights are never committed.

## Development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
