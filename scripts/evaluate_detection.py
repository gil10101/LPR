#!/usr/bin/env python3
"""Evaluate the plate detector on a real VOC-annotated dataset.

Example (RobertLucian/license-plate-dataset layout):
    git clone --depth 1 https://github.com/RobertLucian/license-plate-dataset \
        data/real_robertlucian
    python scripts/evaluate_detection.py \
        --images data/real_robertlucian/dataset/valid/images \
        --annots data/real_robertlucian/dataset/valid/annots
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.eval.detection_eval import evaluate_detection


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, help="Directory of images.")
    ap.add_argument("--annots", required=True, help="Directory of VOC XML files.")
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=1,
                    help="Max predictions kept per image (1 = top box only).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    result = evaluate_detection(args.images, args.annots, iou_threshold=args.iou,
                                max_detections=args.max_det, limit=args.limit)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
