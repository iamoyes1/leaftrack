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

## Datasets

- [CVPPP Leaf Segmentation Challenge](https://www.plant-phenotyping.org/datasets-home)
- [KOMATSUNA](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) time-lapse dataset

See [data/README.md](data/README.md) for where to place them. Datasets and model weights are never committed.

## Development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
