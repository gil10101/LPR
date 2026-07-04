"""Train the CRNN recognizer from scratch with a CTC objective.

Run via ``scripts/train_recognizer.py`` (a thin CLI wrapper). This module holds
the actual loop so it can also be called programmatically (e.g. from tests).

What it does each run:
  * builds train/val loaders from a ``labels.csv`` directory,
  * trains the CRNN with ``nn.CTCLoss``,
  * evaluates exact-match accuracy + CER on val every epoch,
  * keeps the best checkpoint by val accuracy (early stopping),
  * writes a JSON training history (loss/accuracy/CER per epoch) to the reports
    dir so the dashboard can plot learning curves.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..charset import DEFAULT_CODEC, normalize_plate_text
from ..config import Config, load_config
from ..data.recognition_dataset import (PlateRecognitionDataset, ctc_collate,
                                        load_samples_from_labels_csv)
from ..eval.metrics import compute_recognition_metrics
from ..models.crnn import build_crnn_from_config
from ..utils.ctc import greedy_decode


def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, codec, device) -> tuple:
    """Run the model over a loader, returning (metrics, preds, truths)."""
    model.eval()
    preds: List[str] = []
    truths: List[str] = []
    confs: List[float] = []
    for images, _, _, texts in loader:
        images = images.to(device)
        log_probs = model(images)
        for (text, conf), truth in zip(greedy_decode(log_probs, codec), texts):
            preds.append(text)
            truths.append(normalize_plate_text(truth))
            confs.append(conf)
    metrics = compute_recognition_metrics(preds, truths, confs)
    return metrics, preds, truths


def train(config_path: Optional[str] = None,
          data_dir: Optional[str] = None,
          epochs: Optional[int] = None,
          limit: Optional[int] = None) -> dict:
    """Train the recognizer and return a summary dict.

    Parameters
    ----------
    config_path : path to config.yaml (defaults to project root).
    data_dir    : directory containing labels.csv (defaults to synthetic out_dir).
    epochs      : override config epoch count.
    limit       : optional cap on #train samples (fast smoke runs).
    """
    cfg: Config = load_config(config_path)
    codec = DEFAULT_CODEC
    device = resolve_device(cfg.training.device)
    torch.manual_seed(cfg.synthetic.seed)

    data_dir = data_dir or cfg.abspath(cfg.synthetic.out_dir)
    labels_csv = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(
            f"No labels.csv in {data_dir}. Generate data first "
            f"(scripts/generate_synthetic.py) or point --data-dir at a dataset."
        )

    train_samples = load_samples_from_labels_csv(labels_csv, split="train")
    val_samples = load_samples_from_labels_csv(labels_csv, split="val")
    # Only keep samples that actually carry a usable label.
    train_samples = [(p, t) for p, t in train_samples if normalize_plate_text(t)]
    val_samples = [(p, t) for p, t in val_samples if normalize_plate_text(t)]
    if limit:
        train_samples = train_samples[:limit]
    if not train_samples:
        raise RuntimeError("No labelled training samples found.")
    if not val_samples:
        # Carve a val split out of train if the dataset didn't provide one.
        split = max(1, int(0.1 * len(train_samples)))
        val_samples, train_samples = train_samples[:split], train_samples[split:]

    r = cfg.recognizer
    train_ds = PlateRecognitionDataset(train_samples, codec, r.img_height,
                                       r.img_width, r.channels,
                                       augment=cfg.training.augment)
    val_ds = PlateRecognitionDataset(val_samples, codec, r.img_height,
                                     r.img_width, r.channels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=cfg.training.num_workers,
                              collate_fn=ctc_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,
                            shuffle=False, num_workers=cfg.training.num_workers,
                            collate_fn=ctc_collate)

    model = build_crnn_from_config(cfg, codec.num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,
                                 weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    epochs = epochs or cfg.training.epochs
    weights_path = cfg.abspath(r.weights_path)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    reports_dir = cfg.abspath(cfg.training.reports_dir)
    os.makedirs(reports_dir, exist_ok=True)

    print(f"[train] device={device} params={model.num_parameters():,} "
          f"train={len(train_ds)} val={len(val_ds)} epochs={epochs}")

    history: List[dict] = []
    best_acc = -1.0
    best_cer = float("inf")   # early stopping tracks CER, not accuracy
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.time()
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            log_probs = model(images)                     # (T, B, C)
            T = log_probs.size(0)
            input_lengths = torch.full((images.size(0),), T, dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

            running += loss.item()
            n_batches += 1

        train_loss = running / max(1, n_batches)
        metrics, _, _ = evaluate(model, val_loader, codec, device)
        scheduler.step(metrics.exact_match_accuracy)
        dt = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_exact_match_accuracy": metrics.exact_match_accuracy,
            "val_character_accuracy": metrics.character_accuracy,
            "val_cer": metrics.character_error_rate,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": dt,
        })
        print(f"[epoch {epoch:02d}/{epochs}] loss={train_loss:.4f} "
              f"val_acc={metrics.exact_match_accuracy:.4f} "
              f"val_cer={metrics.character_error_rate:.4f} ({dt:.1f}s)")

        # Persist training history every epoch so a dashboard can watch live.
        with open(os.path.join(reports_dir, "training_history.json"), "w") as f:
            json.dump({"history": history, "config": {
                "epochs": epochs, "batch_size": cfg.training.batch_size,
                "lr": cfg.training.lr, "params": model.num_parameters(),
                "train_samples": len(train_ds), "val_samples": len(val_ds),
            }}, f, indent=2)

        # Checkpoint whenever exact-match accuracy improves (the metric we
        # ultimately care about).
        if metrics.exact_match_accuracy > best_acc:
            best_acc = metrics.exact_match_accuracy
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "num_classes": codec.num_classes,
                    "img_height": r.img_height, "img_width": r.img_width,
                    "channels": r.channels, "rnn_hidden": r.rnn_hidden,
                    "rnn_layers": r.rnn_layers, "dropout": r.cnn_dropout,
                },
                "alphabet": codec.alphabet,
                "val_accuracy": best_acc,
            }, weights_path)

        # Early stopping is gated on CER, which decreases steadily even while
        # accuracy is pinned at 0 during the initial CTC blank-collapse phase.
        # Using accuracy here would kill promising runs before they break out.
        if metrics.character_error_rate < best_cer - 1e-4:
            best_cer = metrics.character_error_rate
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.early_stop_patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(val CER stalled for {epochs_no_improve} epochs)")
                break

    total_time = time.time() - start
    summary = {
        "best_val_accuracy": best_acc,
        "epochs_run": len(history),
        "total_seconds": total_time,
        "weights_path": weights_path,
        "params": model.num_parameters(),
    }
    print(f"[train] done in {total_time:.1f}s  best_val_acc={best_acc:.4f}")
    return summary
