"""Train a YOLO11-seg leaf segmentation model (stub, Phase 2).

Will be the single entry point used by the Colab / Kaggle launcher notebooks::

    python train.py --config configs/train_yolo11n.yaml \
        --project /content/drive/MyDrive/leaftrack/runs

It will load a YAML config, train with Ultralytics, save checkpoints to the
given (Google Drive) project directory so runs survive notebook resets, support
``--resume``, and optionally push the best weights to the Hugging Face Hub.
"""

if __name__ == "__main__":
    raise SystemExit("train.py is not implemented yet (Phase 2).")
