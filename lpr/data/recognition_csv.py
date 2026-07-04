"""Load a CSV-annotated recognition dataset into the trainer's layout.

Many recognition datasets (e.g. Kaggle's license-plate text-recognition sets)
ship a folder of plate images plus a CSV mapping each image to its plate string
— for example an ``lpr.csv`` with columns like ``image,text`` or
``filename,plate``. Column names vary, so this loader probes the common ones and
lets you override them.

Output is the standard ``{split}/*.png`` + ``labels.csv`` layout, so the result
drops straight into ``scripts/train_recognizer.py`` (as ``--data-dir`` or mixed
in with ``--extra-data-dir``).
"""
from __future__ import annotations

import csv
import os
import random
import shutil
from typing import List, Optional, Tuple

from ..charset import normalize_plate_text

# Candidate column names, most specific first. Both singular and plural forms
# appear in the wild (e.g. Kaggle's lpr.csv uses "images"/"labels").
_IMAGE_COLUMNS = ["image", "images", "filename", "file", "path", "img",
                  "image_path", "img_path", "image_name", "image_id", "id"]
_TEXT_COLUMNS = ["text", "plate", "label", "labels", "plate_number", "lp",
                 "number", "plate_text", "value", "gt", "ground_truth"]


def _pick_column(header: List[str], candidates: List[str]) -> Optional[str]:
    lower = {h.lower(): h for h in header}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _resolve_image_path(images_dir: str, value: str) -> Optional[str]:
    """Find the image file for a CSV cell that may be a name, id, or rel-path."""
    # Direct path (absolute or relative to images_dir).
    for candidate in (value, os.path.join(images_dir, value)):
        if os.path.isfile(candidate):
            return candidate
    # Bare id/name without extension — try common image extensions.
    stem = os.path.join(images_dir, value)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        if os.path.isfile(stem + ext):
            return stem + ext
    return None


def build_from_recognition_csv(csv_path: str, images_dir: str, out_dir: str,
                               image_column: Optional[str] = None,
                               text_column: Optional[str] = None,
                               val_fraction: float = 0.1,
                               test_fraction: float = 0.1,
                               seed: int = 1234) -> dict:
    """Convert a (image, text) CSV + image folder into a labels.csv dataset.

    Parameters
    ----------
    csv_path      : the annotation CSV (e.g. lpr.csv).
    images_dir    : folder containing the plate images the CSV references.
    out_dir       : destination for the {split}/ crops and labels.csv.
    image_column  : override the auto-detected image column name.
    text_column   : override the auto-detected text column name.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        img_col = image_column or _pick_column(header, _IMAGE_COLUMNS)
        txt_col = text_column or _pick_column(header, _TEXT_COLUMNS)
        if not img_col or not txt_col:
            raise ValueError(
                f"Could not identify image/text columns in {header}. "
                f"Pass image_column/text_column explicitly."
            )
        rows = [(r[img_col], r[txt_col]) for r in reader]

    # Resolve images and keep only rows with a usable label + existing file.
    samples: List[Tuple[str, str]] = []
    missing = 0
    for img_val, text in rows:
        norm = normalize_plate_text(text or "")
        if not norm:
            continue
        path = _resolve_image_path(images_dir, img_val)
        if path is None:
            missing += 1
            continue
        samples.append((path, norm))

    if not samples:
        raise RuntimeError(
            f"No usable (image, text) pairs found. Checked {len(rows)} rows, "
            f"{missing} images missing under {images_dir}."
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)

    def split_for(i: int) -> str:
        if i < n_test:
            return "test"
        if i < n_test + n_val:
            return "val"
        return "train"

    os.makedirs(out_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, s), exist_ok=True)

    lines = ["filepath,text,split"]
    counts = {"train": 0, "val": 0, "test": 0}
    for i, (src, text) in enumerate(samples):
        split = split_for(i)
        fname = f"{i:06d}.png"
        # Copy (or transcode) into the split folder. Keep it simple: copy bytes if
        # already a PNG, else re-encode via OpenCV to normalise the format.
        dst = os.path.join(out_dir, split, fname)
        if src.lower().endswith(".png"):
            shutil.copyfile(src, dst)
        else:
            import cv2
            img = cv2.imread(src)
            if img is None:
                continue
            cv2.imwrite(dst, img)
        lines.append(f"{split}/{fname},{text},{split}")
        counts[split] += 1

    with open(os.path.join(out_dir, "labels.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"out_dir": out_dir, "counts": counts, "total": n,
            "missing_images": missing, "image_column": img_col,
            "text_column": txt_col}
