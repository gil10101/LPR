"""Evaluate a trained recognizer on a test split and export KPIs + charts.

Produces three artefacts consumed by the dashboard and BI tools:
  * ``models/recognizer/metrics.json`` — the full metric bundle (accuracy, CER,
    confusions, calibration, latency) plus example predictions.
  * ``reports/predictions.csv`` — per-sample truth/pred/correct/edit-distance,
    ready to drop into Power BI / Tableau.
  * ``reports/*.png`` — static charts (learning curve, per-length accuracy,
    confusion, calibration) for slide decks / the dashboard fallback.
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..charset import DEFAULT_CODEC, normalize_plate_text
from ..config import Config, load_config
from ..data.recognition_dataset import (PlateRecognitionDataset, ctc_collate,
                                        load_samples_from_labels_csv)
from ..inference.pipeline import PlateRecognizer
from .metrics import compute_recognition_metrics


def evaluate_recognizer(config_path: Optional[str] = None,
                        data_dir: Optional[str] = None,
                        split: str = "test",
                        make_charts: bool = True,
                        num_examples: int = 24) -> dict:
    """Full evaluation pass. Returns the metrics dict and writes artefacts."""
    cfg: Config = load_config(config_path)
    codec = DEFAULT_CODEC
    data_dir = data_dir or cfg.abspath(cfg.synthetic.out_dir)
    labels_csv = os.path.join(data_dir, "labels.csv")
    samples = load_samples_from_labels_csv(labels_csv, split=split)
    samples = [(p, t) for p, t in samples if normalize_plate_text(t)]
    if not samples:
        raise RuntimeError(f"No labelled '{split}' samples in {data_dir}")

    weights = cfg.abspath(cfg.recognizer.weights_path)
    recognizer = PlateRecognizer(weights)

    r = cfg.recognizer
    ds = PlateRecognitionDataset(samples, codec, r.img_height, r.img_width,
                                 r.channels, augment=False)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False,
                        collate_fn=ctc_collate)

    preds: List[str] = []
    truths: List[str] = []
    confs: List[float] = []
    latencies: List[float] = []

    for images, _, _, texts in loader:
        crops_np = _tensors_to_crops(images)
        t0 = time.time()
        decoded = recognizer.read_batch(crops_np)
        dt_ms = (time.time() - t0) * 1000
        latencies.append(dt_ms / max(1, len(crops_np)))
        for (text, conf), truth in zip(decoded, texts):
            preds.append(text)
            truths.append(normalize_plate_text(truth))
            confs.append(conf)

    metrics = compute_recognition_metrics(preds, truths, confs)

    # Latency / throughput KPIs.
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    kpis = metrics.to_dict()
    kpis.update({
        "split": split,
        "mean_latency_ms_per_plate": round(mean_latency, 3),
        "throughput_plates_per_sec": round(1000.0 / mean_latency, 1) if mean_latency else 0.0,
        "model_params": recognizer.model.num_parameters(),
        "device": str(recognizer.device),
        "weights_val_accuracy": recognizer.val_accuracy,
    })

    # A gallery of example predictions for the dashboard. Copy the crops into a
    # committed folder (reports/examples/) so the gallery works on a fresh clone
    # even though the raw dataset (data/) is gitignored. Include a mix of correct
    # and incorrect cases so the gallery is informative, not just a victory lap.
    import shutil
    examples_dir = os.path.join(cfg.abspath(cfg.training.reports_dir), "examples")
    if os.path.isdir(examples_dir):
        for old in os.listdir(examples_dir):
            os.remove(os.path.join(examples_dir, old))
    os.makedirs(examples_dir, exist_ok=True)

    order = sorted(range(len(samples)),
                   key=lambda i: (preds[i] == truths[i]))  # errors first
    chosen = order[:num_examples]
    examples = []
    for rank, i in enumerate(chosen):
        dst_name = f"{rank:03d}.png"
        try:
            shutil.copyfile(samples[i][0], os.path.join(examples_dir, dst_name))
        except OSError:
            continue
        examples.append({
            "image": f"reports/examples/{dst_name}",
            "truth": truths[i], "pred": preds[i],
            "confidence": round(confs[i], 3),
            "correct": preds[i] == truths[i],
        })
    kpis["examples"] = examples

    # Write metrics.json (dashboard reads this).
    metrics_path = cfg.abspath(cfg.recognizer.metrics_path)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(kpis, f, indent=2)

    # Per-sample CSV for BI tools.
    reports_dir = cfg.abspath(cfg.training.reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["truth", "prediction", "correct", "edit_distance", "confidence"])
        from .metrics import _edit_distance
        for pred, truth, conf in zip(preds, truths, confs):
            w.writerow([truth, pred, int(pred == truth),
                        _edit_distance(pred, truth), round(conf, 4)])

    if make_charts:
        try:
            _render_charts(cfg, metrics, kpis, reports_dir)
        except Exception as exc:  # charts are best-effort
            print(f"[eval] chart rendering skipped: {exc}")

    print(f"[eval] split={split} n={metrics.num_samples} "
          f"acc={metrics.exact_match_accuracy:.4f} cer={metrics.character_error_rate:.4f} "
          f"latency={mean_latency:.2f}ms")
    return kpis


def _tensors_to_crops(images: torch.Tensor) -> List[np.ndarray]:
    """Undo preprocessing back to uint8 crops so we time the *real* path.

    The recognizer's ``read_batch`` re-preprocesses, so we hand it image-shaped
    arrays. This keeps the eval path identical to the serving path.
    """
    arr = images.numpy()
    crops = []
    for img in arr:
        img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if img.shape[0] == 1:
            crops.append(img[0])                       # (H, W) grayscale
        else:
            crops.append(np.transpose(img, (1, 2, 0)))  # (H, W, 3)
    return crops


def _render_charts(cfg, metrics, kpis, reports_dir: str) -> None:
    """Render static PNG charts for decks and dashboard fallback."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Learning curve from training history (if present).
    hist_path = os.path.join(reports_dir, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f).get("history", [])
        if hist:
            epochs = [h["epoch"] for h in hist]
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(epochs, [h["train_loss"] for h in hist], "tab:red", label="train loss")
            ax1.set_xlabel("epoch"); ax1.set_ylabel("CTC loss", color="tab:red")
            ax2 = ax1.twinx()
            ax2.plot(epochs, [h["val_exact_match_accuracy"] for h in hist],
                     "tab:blue", label="val accuracy")
            ax2.set_ylabel("val exact-match acc", color="tab:blue")
            ax2.set_ylim(0, 1)
            fig.suptitle("Training curve")
            fig.tight_layout()
            fig.savefig(os.path.join(reports_dir, "learning_curve.png"), dpi=110)
            plt.close(fig)

    # Accuracy by plate length.
    abl = metrics.accuracy_by_length
    if abl:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([str(k) for k in abl], list(abl.values()), color="tab:green")
        ax.set_ylim(0, 1); ax.set_xlabel("plate length"); ax.set_ylabel("accuracy")
        ax.set_title("Exact-match accuracy by length")
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, "accuracy_by_length.png"), dpi=110)
        plt.close(fig)

    # Confidence calibration.
    cal = metrics.calibration
    if cal:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
        ax.plot([c["mean_confidence"] for c in cal],
                [c["accuracy"] for c in cal], "o-", color="tab:purple")
        ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy")
        ax.set_title("Confidence calibration"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, "calibration.png"), dpi=110)
        plt.close(fig)
