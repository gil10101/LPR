"""Convert the OpenALPR benchmark into the recognizer's labels.csv layout.

The OpenALPR benchmark (github.com/openalpr/benchmarks) ships real plate photos
with ground-truth strings in two shapes we can use:

  * ``seg_and_ocr/`` — already-cropped plate images plus a ``groundtruth.csv``
    of ``filename,state,PLATE_TEXT``. These are the cleanest recognition samples.
  * ``endtoend/<region>/`` — full scene photos with a per-image ``.txt`` of
    ``filename x y w h PLATE_TEXT``; we crop the plate box out.

Both are written into the same ``{split}/*.png`` + ``labels.csv`` structure the
trainer and evaluator already read, so real data drops straight into the
pipeline. US plates use the Latin-alphanumeric charset in ``lpr/charset.py``.
"""
from __future__ import annotations

import csv
import os
import random
from typing import List, Tuple

import cv2

from ..charset import normalize_plate_text


def _load_seg_and_ocr(root: str) -> List[Tuple[str, str]]:
    """Return (image_path, text) pairs from the seg_and_ocr crops."""
    gt = os.path.join(root, "seg_and_ocr", "groundtruth.csv")
    img_dir = os.path.join(root, "seg_and_ocr", "usimages")
    pairs: List[Tuple[str, str]] = []
    if not os.path.exists(gt):
        return pairs
    with open(gt) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            fname, _state, text = row[0], row[1], row[2]
            path = os.path.join(img_dir, fname)
            if os.path.exists(path) and normalize_plate_text(text):
                pairs.append((path, text))
    return pairs


def _load_endtoend(root: str, region: str) -> List[Tuple[str, str, tuple]]:
    """Return (image_path, text, (x,y,w,h)) from an endtoend region folder."""
    region_dir = os.path.join(root, "endtoend", region)
    out: List[Tuple[str, str, tuple]] = []
    if not os.path.isdir(region_dir):
        return out
    for fname in os.listdir(region_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(region_dir, fname)) as f:
            line = f.readline().strip()
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 6:
            continue
        img_name = parts[0]
        try:
            x, y, w, h = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        except ValueError:
            continue
        text = parts[5]
        img_path = os.path.join(region_dir, img_name)
        if os.path.exists(img_path) and normalize_plate_text(text):
            out.append((img_path, text, (x, y, w, h)))
    return out


def build_recognition_dataset(benchmark_root: str, out_dir: str,
                              test_fraction: float = 0.2,
                              endtoend_regions: Tuple[str, ...] = ("us", "eu", "br"),
                              seed: int = 1234) -> dict:
    """Materialise real plate crops into ``out_dir`` with a train/test split.

    seg_and_ocr crops (US) are copied directly; endtoend plates from each named
    region are cropped from the scene by their ground-truth box. All regions here
    use Latin-alphanumeric plates, matching the recognizer's charset. Returns a
    summary dict.
    """
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    samples: List[Tuple[str, str]] = []  # (written_relpath, text)
    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Gather crops as (image_bgr, text) so both sources are handled uniformly.
    crops: List[Tuple["cv2.Mat", str]] = []
    for path, text in _load_seg_and_ocr(benchmark_root):
        img = cv2.imread(path)
        if img is not None:
            crops.append((img, text))
    for region in endtoend_regions:
        for path, text, (x, y, w, h) in _load_endtoend(benchmark_root, region):
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 - x1 >= 8 and y2 - y1 >= 6:
                crops.append((img[y1:y2, x1:x2].copy(), text))

    rng.shuffle(crops)
    n_test = int(len(crops) * test_fraction)
    rows = ["filepath,text,split"]
    for i, (img, text) in enumerate(crops):
        split = "test" if i < n_test else "train"
        fname = f"{i:05d}.png"
        cv2.imwrite(os.path.join(out_dir, split, fname), img)
        norm = normalize_plate_text(text)
        rows.append(f"{split}/{fname},{norm},{split}")
        samples.append((f"{split}/{fname}", norm))

    with open(os.path.join(out_dir, "labels.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    n_train = len(crops) - n_test
    return {"out_dir": out_dir, "total": len(crops),
            "train": n_train, "test": n_test}
