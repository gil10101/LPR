#!/usr/bin/env python3
"""Fine-tune a YOLO plate detector on a YOLO-format detection dataset.

For datasets shaped as images/{train,val} + labels/{train,val} — e.g. Kaggle's
large-license-plate detection set:

    pip install ultralytics
    python scripts/train_detector.py --dataset-root /path/to/large-license-plate-dataset

By default it looks for images/train and images/val under the dataset root; pass
--train-images / --val-images if your split folders are named differently. The
best weights are installed to models/detector/plate_yolo.pt; then set
`detector.backend: yolo` in config.yaml to use the learned detector.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.training.train_detector import train_detector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--train-images", default="images/train")
    ap.add_argument("--val-images", default="images/val")
    ap.add_argument("--class-name", default="license-plate")
    ap.add_argument("--base-weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU.")
    args = ap.parse_args()

    summary = train_detector(
        args.dataset_root, train_images=args.train_images,
        val_images=args.val_images, class_name=args.class_name,
        base_weights=args.base_weights, epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=args.device)
    print(f"[detector] done: {summary}")


if __name__ == "__main__":
    main()
