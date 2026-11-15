# Configs

- `splits/komatsuna.json`: the committed plant-level train/val/test split.
  `data/scripts/split_dataset.py` reuses it so every machine trains and
  evaluates on the same plants (pass `--resplit` to draw a new one).
- The Ultralytics `dataset.yaml` is generated next to the processed data
  (it contains a machine-specific absolute path, so it is not committed).
- Training hyperparameter configs arrive in Phase 2.
