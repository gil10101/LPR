# Training locally on your own datasets

This repo is built so large datasets are trained on your machine (ideally with a
GPU), not in a sandbox. Clone it, drop a dataset in, and the scripts handle the
rest.

```bash
git clone https://github.com/gil10101/LPR.git /Users/gil/Stuff/LPR
cd /Users/gil/Stuff/LPR
pip install -r requirements.txt
```

Set `training.device: cuda` in `config.yaml` if you have a GPU (the default
`auto` picks it up automatically).

## Recognizer — a CSV recognition set (image + plate text)

For a dataset shaped as an image folder + a CSV of `(image, text)` — e.g. the
Kaggle "license-plate-text-recognition-dataset" with `lpr.csv`:

```bash
# 1. Convert it into the trainer's layout (columns are auto-detected).
python scripts/prepare_recognition_csv.py \
    --csv /path/to/lpr.csv --images /path/to/images \
    --out-dir data/kaggle_recognition

# 2a. Train the reader on it directly:
python scripts/train_recognizer.py --data-dir data/kaggle_recognition

# 2b. …or mix it with the synthetic set (recommended — real optics + volume):
python scripts/generate_synthetic.py
python scripts/train_recognizer.py \
    --extra-data-dir data/kaggle_recognition --extra-oversample 2

# 3. Evaluate on the real test split.
python scripts/evaluate.py --data-dir data/kaggle_recognition --split test
```

If the CSV uses unusual column names, pass `--image-column` / `--text-column`.

## Detector — a YOLO detection set (images + boxes)

For a YOLO-format set (`images/{train,val}` + `labels/{train,val}`) — e.g. the
Kaggle "large-license-plate-dataset":

```bash
pip install ultralytics

python scripts/train_detector.py \
    --dataset-root /path/to/large-license-plate-dataset \
    --epochs 50 --device 0            # --device 0 = first GPU, or cpu

# The best weights install to models/detector/plate_yolo.pt.
# Switch the app/pipeline to the learned detector:
#   in config.yaml set  detector.backend: yolo
```

## End result

With both trained, the pipeline runs a learned YOLO detector into the CRNN
reader — a real-world ANPR stack. Launch the dashboard to try it:

```bash
python app/app.py            # http://127.0.0.1:5000
```

## Getting big data to a cloud/sandbox run

A cloud agent can't read your local disk. To have one process a dataset, put it
somewhere the run can reach: a GitHub repo under your account (small/medium
sets, or via git-lfs), a release asset, or an object-storage URL. Kaggle and
HuggingFace downloads happen on a machine that can reach them, then the result is
pushed to that reachable location.
