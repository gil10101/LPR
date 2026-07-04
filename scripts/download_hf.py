#!/usr/bin/env python3
"""Download a real license-plate dataset from HuggingFace into local crops.

Requires network access and the optional ``datasets`` package
(``pip install datasets``). The output directory mirrors the synthetic layout
(labels.csv + per-split PNG crops), so training/evaluation work unchanged:

    python scripts/download_hf.py --out-dir data/hf_recognition
    python scripts/train_recognizer.py --data-dir data/hf_recognition

Note: many plate *detection* datasets ship bounding boxes but no plate text; the
recognizer needs text labels. Pick a recognition dataset (image + string) for
training the reader, and a detection dataset for the detector.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.config import load_config
from lpr.data.hf_datasets import download_recognition_dataset


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=cfg.huggingface.recognition_dataset)
    ap.add_argument("--config-name", default=cfg.huggingface.recognition_config)
    ap.add_argument("--out-dir", default=cfg.abspath("data/hf_recognition"))
    ap.add_argument("--cache-dir", default=cfg.abspath(cfg.huggingface.cache_dir))
    ap.add_argument("--max-per-split", type=int, default=None)
    args = ap.parse_args()

    summary = download_recognition_dataset(
        args.dataset, args.out_dir, config=args.config_name,
        cache_dir=args.cache_dir, max_per_split=args.max_per_split)
    print(f"[hf] done: {summary}")
    if summary.get("samples_with_text", 0) == 0:
        print("[hf] WARNING: no text labels found in this dataset — it is likely "
              "a detection-only dataset. Use it for the detector, not the reader.")


if __name__ == "__main__":
    main()
