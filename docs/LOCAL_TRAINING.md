# Training locally on your own datasets (GPU)

Large datasets train on your machine, not in a sandbox. The steps below are the
full runbook for an NVIDIA/CUDA box. `config.yaml` ships GPU-sized defaults
(`batch_size: 256`, `num_workers: 8`); the device is `auto`, which selects CUDA
automatically.

## 0. Set up the machine (once)

```bash
git clone https://github.com/gil10101/LPR.git
cd LPR
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build (or cu124)
pip install ultralytics                                                # for the detector
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"   # must be True
```

## 1. Get the datasets onto this machine

Put both under `data/` (gitignored). Set your Kaggle credentials once in a
`.env` file at the repo root, then run the download script:

```bash
pip install kaggle
cp .env.example .env          # Windows: copy .env.example .env
#   edit .env and set KAGGLE_USERNAME + KAGGLE_KEY
#   (kaggle.com -> Settings -> API -> Create New Token gives you both)
python scripts/download_kaggle.py
```

`.env` is gitignored, so your key is never committed. The script loads the
credentials before importing the Kaggle client (which avoids the
`KeyError: 'username'` from a half-configured token) and unzips both sets into
`data/nicklpsr` and `data/llpd`.

Then confirm the paths — the recognizer needs `lpr.csv` + the crops folder, the
detector needs `images/{train,val}` + `labels/{train,val}`:

```bash
ls data/nicklpsr           # expect: lpr.csv  cropped_lps/
ls data/llpd/images        # expect: train  val  (test)
```

Adjust the `--csv/--images/--dataset-root` flags below if your unzip puts them
elsewhere.

## 2. Recognizer  (~5–10 min on GPU)

```bash
python scripts/prepare_recognition_csv.py \
    --csv data/nicklpsr/lpr.csv --images data/nicklpsr/cropped_lps \
    --out-dir data/kaggle_recognition

python scripts/train_recognizer.py --data-dir data/kaggle_recognition
python scripts/evaluate.py --data-dir data/kaggle_recognition --split test
```

Optional — mix in synthetic for extra robustness:
```bash
python scripts/generate_synthetic.py
python scripts/train_recognizer.py \
    --extra-data-dir data/kaggle_recognition --extra-oversample 4
```

## 3. Detector  (~15–35 min on GPU)

```bash
python scripts/train_detector.py \
    --dataset-root data/llpd --device 0 --epochs 50 --batch 48 \
    --base-weights yolov8s.pt --imgsz 640
```

That installs `models/detector/plate_yolo.pt`. Turn it on:
```yaml
# config.yaml
detector:
  backend: yolo
```

## 4. Run the app

```bash
python app/app.py            # http://127.0.0.1:5000
```

Upload a car photo: the YOLO detector finds the plate, the CRNN reads it, and the
`/dashboard` page shows the KPIs from your evaluation.

## Notes

- `evaluate.py` writes `reports/*.png` charts and updates `models/recognizer/
  metrics.json`, which the dashboard reads. Re-run it after any training.
- Both trainers save the best checkpoint continuously, so interrupting a run
  (Ctrl-C) still leaves a usable model.
- Training on CPU instead? Lower `batch_size` to 64–128 in `config.yaml`.
