"""Evaluate the plate detector on a real, boxed dataset (PASCAL VOC format).

The synthetic pipeline validates recognition; this validates *detection* on real
photographs. It reads a folder of images + VOC XML annotations (as shipped by the
RobertLucian/license-plate-dataset and many other ANPR sets), runs the detector,
and reports precision / recall / F1 / mean-IoU at a chosen IoU threshold — real
KPIs, not synthetic ones.

Detection datasets rarely carry the plate *text*, so this measures localisation
only; recognition accuracy is measured separately on labelled crops.
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import cv2

from ..config import load_config
from ..models.detector import build_detector
from .metrics import detection_prf


def parse_voc_boxes(xml_path: str) -> List[List[float]]:
    """Return [x1,y1,x2,y2] boxes for every object in a VOC annotation file."""
    boxes: List[List[float]] = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return boxes
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            boxes.append([
                float(bnd.findtext("xmin")), float(bnd.findtext("ymin")),
                float(bnd.findtext("xmax")), float(bnd.findtext("ymax")),
            ])
        except (TypeError, ValueError):
            continue
    return boxes


def _find_pairs(images_dir: str, annots_dir: str) -> List[Tuple[str, str]]:
    """Match each image to its VOC annotation by shared basename."""
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(images_dir, "*"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(annots_dir, stem + ".xml")
        if os.path.exists(xml_path):
            pairs.append((img_path, xml_path))
    return pairs


def evaluate_detection(images_dir: str, annots_dir: str,
                       config_path: Optional[str] = None,
                       iou_threshold: float = 0.4,
                       max_detections: int = 1,
                       limit: Optional[int] = None) -> dict:
    """Run the configured detector over a VOC dataset and return detection KPIs.

    ``max_detections`` caps predictions per image. Most VOC plate sets have one
    plate per frame, so the default of 1 (keep only the top-scoring box) gives a
    fair precision; raise it for multi-plate scenes.
    """
    cfg = load_config(config_path)
    detector = build_detector(cfg)
    pairs = _find_pairs(images_dir, annots_dir)
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError(f"No image/annotation pairs under {images_dir}")

    preds_per_image: List[List[list]] = []
    gts_per_image: List[List[list]] = []
    for img_path, xml_path in pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gt = parse_voc_boxes(xml_path)
        dets = detector.detect(img, max_detections=max_detections)
        preds_per_image.append([d.box for d in dets])
        gts_per_image.append(gt)

    prf = detection_prf(preds_per_image, gts_per_image, iou_threshold)
    prf["num_images"] = len(preds_per_image)
    prf["backend"] = cfg.detector.backend
    return prf
