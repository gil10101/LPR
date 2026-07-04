#!/usr/bin/env python3
"""Train the CRNN recognizer from scratch (CTC loss).

Usage:
    python scripts/train_recognizer.py                     # config.yaml settings
    python scripts/train_recognizer.py --epochs 30
    python scripts/train_recognizer.py --data-dir data/hf_recognition
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.training.train_recognizer import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None,
                    help="Directory containing labels.csv (default: synthetic).")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap #train samples (for quick smoke runs).")
    ap.add_argument("--extra-data-dir", default=None,
                    help="A second labels.csv dir (e.g. real plates) to mix in.")
    ap.add_argument("--extra-oversample", type=int, default=1,
                    help="Repeat the extra train samples this many times.")
    args = ap.parse_args()
    train(config_path=args.config, data_dir=args.data_dir,
          epochs=args.epochs, limit=args.limit,
          extra_data_dir=args.extra_data_dir,
          extra_oversample=args.extra_oversample)


if __name__ == "__main__":
    main()
