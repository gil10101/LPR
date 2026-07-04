# Project Log

Notes on how the system was built, the decisions behind it, the measured
results, and what I'd do next. Written for a reviewer who wants the reasoning,
not just the file list.

## What it is

A two-stage license plate reader. A detector locates the plate in a photo; a
CRNN + CTC model reads the characters off the crop. A Flask app exposes both an
upload/recognize view and a metrics dashboard.

## Build summary

- **Recognizer** (`lpr/models/crnn.py`) — CNN backbone + 2-layer BiLSTM + CTC
  head, ~6.6M params, trained from scratch. CTC was the key choice: it learns the
  character alignment, so training needs only `(crop, text)` pairs.
- **Synthetic data** (`lpr/data/synthetic.py`) — renders plates across layouts,
  colour schemes, and six fonts, with perspective/blur/noise. This supplies the
  text labels the recognizer needs and keeps the pipeline reproducible end to end.
- **Detection** (`lpr/models/detector.py`) — a classical OpenCV detector by
  default (no extra deps), with an optional Ultralytics YOLO backend behind the
  same interface.
- **Pipeline + app** (`lpr/inference/pipeline.py`, `app/`) — detect → recognize →
  annotate, served through a dashboard with a KPI view.
- **Metrics** (`lpr/eval/`) — exact-match accuracy, CER, character accuracy, mean
  edit distance, confidence calibration, character confusions, detection
  precision/recall/IoU, and latency/throughput.

## Results

Held-out 800-plate synthetic test split:

| Metric | Value |
|--------|-------|
| Exact-match accuracy | **0.991** |
| Character accuracy (1 − CER) | 0.999 (CER 0.0013) |
| End-to-end scene accuracy | 0.98 |
| Latency | ~6 ms/plate (~169 plates/sec, CPU) |
| Model size | 6.6M params |
| Top confusion | 0 ↔ O |

Detection on real dashcam frames (`RobertLucian/license-plate-dataset`, classical
backend): mean IoU 0.60 on matched plates. The classical detector handles clear,
close plates; YOLO is the backend for small, distant plates in cluttered footage.

## Decisions that mattered

- **Margin augmentation.** Adding/shaving random borders on training crops made
  the recognizer robust to the detector's imperfect boxes. End-to-end scene
  accuracy went 0.50 → 0.98, and test accuracy 0.978 → 0.991.
- **Early stopping on CER.** CTC training sits in a blank-collapse plateau for
  ~6 epochs (accuracy 0, loss near `log(num_classes)`) before it breaks out.
  Gating early stopping on CER — which keeps dropping through the plateau —
  keeps promising runs alive; gating on accuracy would kill them.
- **Dependency pinning.** torch 2.2 needs numpy < 2 and opencv 4.x; the pins in
  `requirements.txt` are the compatible set.
- **Committing the trained model + KPI report.** The dashboard runs on a fresh
  clone without a training step, which makes it easy to demo and review.

## Data sourcing

The recognizer needs plate-text labels, which the synthetic generator provides.
For real data, the code is ready: `scripts/download_hf.py` loads a HuggingFace
recognition dataset into the trainer's layout, and `scripts/evaluate_detection.py`
scores the detector on any VOC dataset. `docs/DATASETS.md` lists the specific
datasets and commands.

## Next steps

1. **Train the YOLO detector** on the real detection set and switch
   `detector.backend` to `yolo` for cluttered-scene robustness.
2. **Train the recognizer on real, text-labelled plates** (e.g. CCPD) and blend
   with synthetic to keep clean-crop accuracy while generalising to real optics.
3. **Beam-search decoding with plate-format priors** to squeeze the last few
   percent and handle 0/O and I/1 systematically.
4. **Video/stream support** and a timestamped results store for an E-ZPass-style
   log.
5. **Dockerfile** for one-command deploy.

## Reflection

The synthetic generator was the highest-leverage decision — it made the project
trainable, testable, and demoable in one shot, and margin augmentation on top of
it closed the train/serve gap between clean crops and real detections. The
sharpest surprise was CTC blank collapse: the loss plateaus convincingly enough
to look like a dead run, and the fix was in the training loop's stopping
criterion, not the model. The recognizer is strong on its training distribution;
the clearly-scoped next investments are the learned detector and real-plate
recognition data.
