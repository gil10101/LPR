#!/usr/bin/env python3
"""Evaluate the trained recognizer and export KPIs + charts.

Usage:
    python scripts/evaluate.py                 # test split from config
    python scripts/evaluate.py --split val
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.eval.evaluate import evaluate_recognizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()
    evaluate_recognizer(config_path=args.config, data_dir=args.data_dir,
                        split=args.split, make_charts=not args.no_charts)


if __name__ == "__main__":
    main()
