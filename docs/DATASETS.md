# Datasets

The two stages want different labels, and that drives which dataset feeds which:

- **Detection** trains on images + **bounding boxes** (where the plate is).
- **Recognition** trains on plate crops + the **plate text** (what it reads).

## What trains the recognizer

The synthetic generator (`lpr/data/synthetic.py`). It emits `(crop, text)` pairs
directly, so the reader has ground-truth strings to learn from, and the whole
pipeline stays reproducible with no external downloads. It's what produced the
99.1% test model committed here.

To train the reader on real plates, use a recognition dataset that ships the
plate string and load it with `scripts/download_hf.py`, which writes the same
`labels.csv` + per-split crops layout the trainer already reads. CCPD is a good
choice — its filenames encode the plate number, so labels come for free.

## What validates the detector

Any bounding-box dataset. `lpr/eval/detection_eval.py` scores the detector on a
PASCAL-VOC folder and reports precision / recall / F1 / mean-IoU.

**`RobertLucian/license-plate-dataset`** (GitHub) — 534 real dashcam frames with
VOC boxes. Cloned and evaluated here; results in
`reports/detection_metrics_real.json`.

```bash
git clone --depth 1 https://github.com/RobertLucian/license-plate-dataset data/real
python scripts/evaluate_detection.py \
    --images data/real/dataset/valid/images --annots data/real/dataset/valid/annots
```

## Referenced detection datasets

Both carry bounding boxes and train/validate the detector.

**`keremberke/license-plate-object-detection`** (HuggingFace). The loader is
wired for its COCO-style schema and set as the default in `config.yaml`:

```bash
pip install datasets
python scripts/download_hf.py \
    --dataset keremberke/license-plate-object-detection --config-name full \
    --out-dir data/hf_keremberke
```

**Kaggle — `amirhoseinahmadnejad/car-license-plate-detection-iran`** (Iranian
plates). Needs a Kaggle API token (`~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d amirhoseinahmadnejad/car-license-plate-detection-iran \
    -p data/iran --unzip
python scripts/evaluate_detection.py --images data/iran/images --annots data/iran/annotations
```

## Reading non-Latin plates

The alphabet lives in one place. To read Iranian, EU, or other regional plates,
extend `ALPHABET` in `lpr/charset.py` with the required glyphs and train on
labelled crops for that region. The architecture stays the same — only the output
alphabet and training data change.
