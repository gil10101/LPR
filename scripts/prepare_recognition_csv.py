#!/usr/bin/env python3
"""Convert a CSV-annotated recognition dataset into the trainer's layout.

For datasets shaped like an image folder + a CSV of (image, plate_text) — e.g.
Kaggle's license-plate text-recognition sets with an lpr.csv:

    python scripts/prepare_recognition_csv.py \
        --csv /path/to/lpr.csv --images /path/to/images \
        --out-dir data/kaggle_recognition

    # then train on it (or mix it with synthetic):
    python scripts/train_recognizer.py --data-dir data/kaggle_recognition
    python scripts/train_recognizer.py --extra-data-dir data/kaggle_recognition --extra-oversample 3

Column names are auto-detected; override with --image-column / --text-column if
the CSV uses unusual headers.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.data.recognition_csv import build_from_recognition_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Annotation CSV (e.g. lpr.csv).")
    ap.add_argument("--images", required=True, help="Folder of plate images.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-column", default=None)
    ap.add_argument("--text-column", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    args = ap.parse_args()

    summary = build_from_recognition_csv(
        args.csv, args.images, args.out_dir,
        image_column=args.image_column, text_column=args.text_column,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    print(f"[recognition-csv] done: {summary}")


if __name__ == "__main__":
    main()
