# Notebooks

Thin launchers for free T4 GPUs. Each one only clones this repo, installs it,
prepares the data and calls `train.py` / `evaluate.py`. All logic lives in the
`leaftrack` package.

| Notebook | Where checkpoints go | Notes |
|---|---|---|
| [`train_colab.ipynb`](train_colab.ipynb) | Google Drive (`MyDrive/leaftrack/runs`) | Set `RESUME = True` and re-run after a disconnect. The dataset archives are cached on Drive, so the slow download happens once. |
| [`train_kaggle.ipynb`](train_kaggle.ipynb) | `/kaggle/working` (saved with the notebook version) | Kaggle can't mount Drive; publish the final weights to the Hugging Face Hub. Needs Internet enabled. |

Both can publish `best.pt`, `metrics.json` and a model card to the Hugging
Face Hub when an `HF_TOKEN` secret and `HF_REPO` are set. Weights are never
committed to git.
