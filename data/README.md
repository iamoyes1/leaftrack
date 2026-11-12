# Data

Raw datasets live in `data/raw/` and are **never committed** (see `.gitignore`).

Expected layout once downloaded:

```
data/raw/
├── cvppp/       # CVPPP Leaf Segmentation Challenge
└── komatsuna/   # KOMATSUNA time-lapse dataset
```

Sources:

- CVPPP LSC: https://www.plant-phenotyping.org/datasets-home
- KOMATSUNA: http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/

`data/scripts/` will hold the converter to YOLO-seg polygon format, the
sequence-aware train/val/test split, and the polygon visualization check.
They are written only after the real files have been inspected, because label
formats are read from the data rather than assumed.
