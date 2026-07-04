# License Plate Recognition System

Detects a license plate in a photo and reads its number with a CRNN sequence
model trained from pixels to text. Ships with a web dashboard for uploading
images and a metrics view for the model KPIs.

```
   photo ──▶  [ detection ]  ──▶  plate crop  ──▶  [ CRNN + CTC ]  ──▶  "ABC1234"
            locate the plate                     read the characters
```

## How the recognizer works

The reader is a **CRNN** — a convolutional stack followed by a bidirectional
LSTM — trained end-to-end with a **CTC** loss (Shi et al., 2015). I picked this
architecture because CTC learns the character alignment itself: I only need
`(crop, "ABC1234")` pairs, no per-character boxes or segmentation.

1. A CNN turns a `32×128` grayscale crop into a width-wise feature sequence.
2. A 2-layer BiLSTM adds left/right context along that sequence.
3. A linear head emits per-timestep scores over `[0-9 A-Z] + blank`.
4. CTC aligns the length-`T` prediction to the shorter label during training;
   greedy decoding collapses repeats and drops blanks at inference.

It's ~6.6M parameters and trains from scratch on a laptop CPU in under an hour.

## Results

The shipped model is trained on synthetic plates plus real OpenALPR crops, and
evaluated on a held-out split of each:

| KPI | Synthetic test (3,000) | Real test (239, OpenALPR) |
|-----|:---:|:---:|
| Exact-match accuracy | 99.3% | **91.2%** |
| Character accuracy (1 − CER) | 99.9% | 98.5% |

~7 ms/plate (~150/sec on CPU), 6.6M parameters. Training on synthetic alone
scores ~99% on synthetic but only ~10% on real plates; mixing in real crops is
what closes that gap — see `docs/PROJECT_LOG.md`. The dominant remaining error is
`0` vs `O`, the expected hard case for plate reading.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the training data (offline synthetic generator).
python scripts/generate_synthetic.py

# 2. Train the recognizer.
python scripts/train_recognizer.py         # -> models/recognizer/crnn.pt

# 3. Evaluate: writes metrics.json, charts, and a per-sample CSV.
python scripts/evaluate.py

# 4. Serve the dashboard.
python app/app.py                          # http://127.0.0.1:5000
```

The trained model and KPI report are committed, so `python app/app.py` works
immediately after cloning.

## Dashboard

- **Recognize** (`/`) — drop in a car photo or a plate crop; returns the
  annotated detection, the plate string, per-plate confidence, and stage timings.
- **Metrics & KPIs** (`/dashboard`) — headline KPIs, the training curve,
  accuracy-by-length, a confidence-calibration plot, the top character
  confusions, and a gallery of example predictions. Fed by `/api/metrics`.

## Design decisions worth flagging

- **Synthetic training data.** `lpr/data/synthetic.py` renders plates (multiple
  layouts, colour schemes, six fonts) with perspective, blur, and noise. It's the
  source of the `(crop, text)` pairs the recognizer needs, and it makes the whole
  pipeline — training, evaluation, KPIs — reproducible with no external
  dependencies. It doubles as an augmentation source.
- **Margin augmentation.** Training crops get random borders added or shaved so
  the recognizer tolerates the detector's imperfect boxes. This took end-to-end
  scene accuracy from 50% to 98% and test accuracy from 97.8% to 99.1%.
- **Early stopping on CER.** CTC models sit in a "blank collapse" plateau for the
  first few epochs (accuracy pinned at 0, loss near `log(num_classes)`) before
  breaking out. Early stopping watches CER, which keeps improving through the
  plateau, so runs aren't cut short prematurely.
- **Pluggable detection.** A dependency-free OpenCV detector is the default so the
  app runs anywhere; an Ultralytics YOLO backend (`detector.backend: yolo`) is the
  drop-in upgrade for cluttered scenes.

## Data

The recognizer trains on the synthetic generator, which provides the plate-text
labels it needs. For real data:

- **Detection sets** (boxes) train and evaluate the detector.
  `lpr/eval/detection_eval.py` scores the detector on any PASCAL-VOC dataset:
  ```bash
  git clone --depth 1 https://github.com/RobertLucian/license-plate-dataset \
      data/real_robertlucian
  python scripts/evaluate_detection.py \
      --images data/real_robertlucian/dataset/valid/images \
      --annots data/real_robertlucian/dataset/valid/annots
  ```
- **Recognition sets** (image + plate text) train the reader.
  `scripts/download_hf.py` pulls a HuggingFace dataset into the same
  `labels.csv` + crops layout the trainer already reads.

`docs/DATASETS.md` covers the specific datasets and download commands;
`docs/LOCAL_TRAINING.md` walks through training on your own Kaggle/HF datasets
(CSV recognition sets and YOLO detection sets) on a GPU machine;
`docs/PROJECT_LOG.md` records the build history, results, and next steps.

## Detection

Two backends behind one interface (`lpr/models/detector.py`):

- **Classical** (default) — OpenCV edges + morphology + contour/aspect-ratio
  filtering. Zero dependencies, localises clear/close plates well.
- **YOLO** (optional) — an Ultralytics model fine-tuned on plate boxes; the
  choice for small, distant plates in cluttered footage. Set
  `detector.backend: yolo` and drop weights at `models/detector/plate_yolo.pt`.

When detection returns nothing, the pipeline treats the upload as an
already-cropped plate, so plate crops can be fed in directly.

## Project layout

```
config.yaml                 # single source of truth for every setting
lpr/
  charset.py                # alphabet + CTC encode/decode
  config.py                 # typed config loader
  data/
    synthetic.py            # synthetic plate generator
    hf_datasets.py          # HuggingFace loaders
    recognition_dataset.py  # torch Dataset + CTC collate + preprocessing
  models/
    crnn.py                 # CRNN (CNN + BiLSTM + CTC head)
    detector.py             # classical + YOLO plate detectors
  training/train_recognizer.py   # CTC training loop, early stopping on CER
  eval/
    metrics.py              # CER, accuracy, confusion, calibration, detection PRF
    evaluate.py             # recognizer eval -> metrics.json + CSV + charts
    detection_eval.py       # detector eval on VOC datasets
  inference/pipeline.py     # detect -> recognize -> annotate
  utils/ctc.py              # greedy CTC decode + confidence
app/                        # Flask dashboard (recognize + KPI views)
scripts/                    # generate / train / evaluate / evaluate_detection / download_hf
tests/                      # unit tests for the core logic
docs/                       # dataset notes + project log
```

## Configuration

All settings live in `config.yaml` — model dimensions, training
hyper-parameters, dataset sizes, the detector backend, and the web server. The
scripts and the app read from it, so a change there applies everywhere. For GPU
training, set `training.device: cuda` (it's `auto` by default).

## Requirements

Python 3.9+, PyTorch, OpenCV, Flask, NumPy, Pillow, and matplotlib.
`requirements.txt` uses version ranges, so `pip install -r requirements.txt`
resolves a compatible modern stack on Linux, macOS/Apple Silicon, or CUDA.
