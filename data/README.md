# Data

Raw and processed datasets are **never committed** (see `.gitignore`).

## KOMATSUNA (used)

Top-down time-lapse images of komatsuna plants with per-leaf instance labels
([Uchiyama et al., ICCV Workshops 2017](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/)).

Download the core files into `data/raw/komatsuna/_archives/` and extract them
into `data/raw/komatsuna/`:

| File | Size | Contents |
|---|---|---|
| `multi_plant.zip` | 113 MB | 900 plant crops from 3 overhead cameras |
| `multi_label.zip` | 2.1 MB | 900 matching leaf labels |
| `rgbd_plant.zip` | 9.3 MB | 300 plant crops from an overhead RGB-D camera |
| `rgbd_label.zip` | 0.3 MB | 300 matching leaf labels |
| `multi_readme.pdf`, `rgbd_readme.pdf` | 0.6 MB | file naming and label palette |

What the files actually contain (verified, not assumed):

- **Two rigs, 10 different plants.** `multi`: 5 plants, each seen by 3 cameras
  (`rgb_CC_PP_DDD_TT.png`, 480x480 or 544x480). `rgbd`: 5 other plants, one
  camera (`rgb_PP_DDD_TT.png`, 166x190 or 204x185).
- **Time-lapse.** 10 days x 6 captures/day (every 4 h) = 60 frames per plant
  and camera. Every frame is labelled.
- **Labels.** RGB PNGs, black background, one fixed color per leaf assigned by
  age (blue = oldest, then green, cyan, red, magenta, yellow, gray, navy).
  A leaf keeps its color for the whole sequence, so colors are ground-truth
  track IDs. 2-8 leaves per image. About 5% of leaf masks consist of several
  disconnected pieces (petioles hidden by other leaves). 30 RGB-D labels contain
  1-7 stray white pixels, which are ignored.

## CVPPP (not used)

The CVPPP / Plant Phenotyping Datasets terms forbid commercial use (including
screenshots in advertising) and redistribution, which conflicts with a public
portfolio demo and published weights. It is excluded unless the dataset owners
grant permission.

## Pipeline

```bash
python data/scripts/convert_komatsuna.py   # -> data/processed/komatsuna/{images,labels,manifest.csv}
python data/scripts/split_dataset.py       # -> train/val/test.txt + dataset.yaml
python data/scripts/visualize_labels.py    # -> outputs/check/*.png for a visual check
```

- **Conversion** turns each leaf color into one YOLO-seg polygon (class 0,
  `leaf`). Multi-piece masks are merged into one outline with zero-width
  bridges. `manifest.csv` keeps each label line's ground-truth leaf ID for
  tracking evaluation. Re-rasterised polygons match the original masks with a
  mean IoU of 0.968 (0.982 for leaves of 1000 px or more; the lowest values
  are leaves of only 9-40 px).
- **Splitting** is by physical plant, never by frame: all frames of a plant,
  from every camera, land in one split, and each split contains both rigs.
  The plant assignment is committed in
  [`configs/splits/komatsuna.json`](../configs/splits/komatsuna.json) and
  reused on every machine.

| Split | Plants | Images | Leaf instances |
|---|---|---|---|
| train | multi 01, 02, 03 + rgbd 00, 02, 03 | 720 | 3497 |
| val | multi 04 + rgbd 04 | 240 | 1234 |
| test | multi 00 + rgbd 01 | 240 | 1087 |
