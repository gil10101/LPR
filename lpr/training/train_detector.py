"""Fine-tune an Ultralytics YOLO plate detector on a YOLO-format dataset.

Detection datasets such as Kaggle's large-license-plate set ship YOLO labels
(images/{train,val} + labels/{train,val} with ``class cx cy w h`` normalised
boxes). This wraps the Ultralytics trainer: it writes the ``data.yaml`` the
trainer needs, runs training, and copies the best weights to the path the
inference pipeline loads (``models/detector/plate_yolo.pt``), after which setting
``detector.backend: yolo`` in config.yaml switches the app over.

Ultralytics is an optional dependency (``pip install ultralytics``); training on
a real detection set benefits a lot from a GPU.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

import yaml

from ..config import Config, load_config


def _write_data_yaml(dataset_root: str, out_path: str,
                     train_dir: str, val_dir: str, class_name: str) -> None:
    data = {
        "path": os.path.abspath(dataset_root),
        "train": train_dir,
        "val": val_dir,
        "names": {0: class_name},
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def train_detector(dataset_root: str,
                   config_path: Optional[str] = None,
                   train_images: str = "images/train",
                   val_images: str = "images/val",
                   class_name: str = "license-plate",
                   base_weights: str = "yolov8n.pt",
                   epochs: int = 50,
                   imgsz: int = 640,
                   batch: int = 16,
                   device: Optional[str] = None) -> dict:
    """Train a YOLO detector and install the best weights for inference.

    ``dataset_root`` is the folder that contains the images/ and labels/ trees.
    Returns a summary dict with the path to the installed weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Ultralytics is required to train the detector. "
            "Install it with: pip install ultralytics"
        ) from exc

    cfg: Config = load_config(config_path)
    data_yaml = os.path.join(dataset_root, "lpr_data.yaml")
    _write_data_yaml(dataset_root, data_yaml, train_images, val_images, class_name)

    model = YOLO(base_weights)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                          batch=batch, device=device, project="reports/yolo",
                          name="plate_detector", exist_ok=True)

    # Install the best checkpoint where the pipeline expects it.
    best = os.path.join("reports/yolo/plate_detector/weights/best.pt")
    dest = cfg.abspath(cfg.detector.yolo_weights)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(best):
        shutil.copyfile(best, dest)

    metrics = getattr(results, "results_dict", {}) or {}
    print(f"[detector] trained; weights -> {dest}")
    print(f"[detector] set 'detector.backend: yolo' in config.yaml to use it.")
    return {"weights": dest, "data_yaml": data_yaml, "metrics": metrics}
