"""HuggingFace dataset loaders for real license-plate data.

These are used on machines with network access to `huggingface.co`. In sandboxed
environments (no network) the synthetic generator stands in — the training code
treats both as the same "directory of crops + labels.csv" interface, so nothing
downstream changes.

Two dataset shapes are supported:

  * Object-detection style (images + bbox annotations, e.g. the
    ``keremberke/license-plate-object-detection`` family). We crop each plate
    box out of the scene to build recognition training data. Note: many
    detection datasets do NOT ship the plate *text*, only boxes — in that case
    crops are still useful for the detector but not for the recognizer.

  * Recognition style (plate image + text label). Used directly.

Because dataset schemas vary, the extraction is defensive: it probes common
column names and skips anything it can't interpret rather than crashing.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..charset import normalize_plate_text

# Column names seen in the wild for the plate-text field.
_TEXT_COLUMNS = ["text", "label", "plate", "plate_number", "lp", "number",
                 "ground_truth", "gt", "value"]


def _find_text(example: dict) -> Optional[str]:
    for col in _TEXT_COLUMNS:
        if col in example and isinstance(example[col], str) and example[col].strip():
            return example[col]
    return None


def download_recognition_dataset(dataset_id: str, out_dir: str,
                                 config: Optional[str] = None,
                                 cache_dir: Optional[str] = None,
                                 max_per_split: Optional[int] = None) -> dict:
    """Materialise a HuggingFace dataset into the local crops+labels layout.

    Writes ``out_dir/{split}/*.png`` and ``out_dir/labels.csv`` so the result is
    a drop-in replacement for the synthetic dataset. Requires the ``datasets``
    package and network access; raises a clear error if unavailable.
    """
    try:
        from datasets import load_dataset  # imported lazily; optional dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The 'datasets' package is required for HuggingFace loading. "
            "Install it with: pip install datasets"
        ) from exc

    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(dataset_id, name=config, cache_dir=cache_dir)

    rows = ["filepath,text,split"]
    counts: dict = {}
    kept_with_text = 0

    split_map = {"train": "train", "validation": "val", "valid": "val",
                 "val": "val", "test": "test"}
    for hf_split in ds.keys():
        split = split_map.get(hf_split, hf_split)
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        n = 0
        for i, example in enumerate(ds[hf_split]):
            if max_per_split and i >= max_per_split:
                break
            crops = _extract_crops(example)
            for j, (crop, text) in enumerate(crops):
                fname = f"{i:06d}_{j}.png"
                crop.save(os.path.join(split_dir, fname))
                norm = normalize_plate_text(text) if text else ""
                rows.append(f"{split}/{fname},{norm},{split}")
                if norm:
                    kept_with_text += 1
                n += 1
        counts[split] = n

    with open(os.path.join(out_dir, "labels.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    return {"out_dir": out_dir, "counts": counts,
            "samples_with_text": kept_with_text, "dataset_id": dataset_id}


def _extract_crops(example: dict) -> List[Tuple[Image.Image, Optional[str]]]:
    """Pull plate crops (and text if present) out of one dataset example.

    Handles both "whole image is the plate" recognition rows and
    object-detection rows carrying bounding boxes.
    """
    image = example.get("image")
    if image is None:
        return []
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.asarray(image))
        except Exception:
            return []
    image = image.convert("RGB")

    text = _find_text(example)

    # Detection style: crop each bbox out of the scene.
    objects = example.get("objects")
    if isinstance(objects, dict) and "bbox" in objects:
        bboxes = objects.get("bbox", [])
        crops = []
        for bbox in bboxes:
            crop = _crop_bbox(image, bbox)
            if crop is not None:
                crops.append((crop, text))
        if crops:
            return crops

    # Recognition style: the image itself is the plate.
    return [(image, text)]


def _crop_bbox(image: Image.Image, bbox) -> Optional[Image.Image]:
    """Crop a bbox that may be [x,y,w,h] or [x1,y1,x2,y2]."""
    try:
        x0, y0, a, b = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    W, H = image.size
    # Heuristic: if the 3rd/4th values look like width/height (smaller than the
    # image and a+x < W), treat as xywh; otherwise as xyxy.
    if a <= W and b <= H and (x0 + a) <= W + 1 and (y0 + b) <= H + 1:
        x1, y1, x2, y2 = x0, y0, x0 + a, y0 + b
    else:
        x1, y1, x2, y2 = x0, y0, a, b
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 4:
        return None
    return image.crop((x1, y1, x2, y2))
