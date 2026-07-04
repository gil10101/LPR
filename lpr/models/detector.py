"""Plate detection (localisation) stage.

The recognizer reads a *cropped* plate; this module finds where that crop is
inside a full photo. Two backends share one interface:

  * ClassicalPlateDetector — pure OpenCV (edges + morphology + contour /
    aspect-ratio filtering). Needs no training or extra dependencies, so the
    web app works end-to-end offline. Good for reasonably framed photos.

  * YoloPlateDetector — wraps an Ultralytics YOLO model fine-tuned on plate
    boxes. Far more robust on cluttered scenes, but requires the optional
    ``ultralytics`` package and a trained weights file.

Both return a list of ``Detection`` objects with pixel boxes and a score. If a
detector finds nothing, callers fall back to treating the whole image as the
plate (common when the user uploads an already-cropped plate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """A localised plate: pixel box, confidence and the cropped image."""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    crop: np.ndarray  # BGR crop of the plate region

    @property
    def box(self):
        return [self.x1, self.y1, self.x2, self.y2]


class ClassicalPlateDetector:
    """Heuristic plate localiser using classical computer vision.

    Pipeline: grayscale -> bilateral filter (denoise, keep edges) -> Sobel/Canny
    edges -> morphological close (join characters into a plate-shaped blob) ->
    contours -> filter by area and plate-like aspect ratio. It is deliberately
    simple and dependency-free; the learned YOLO backend is the upgrade path.
    """

    def __init__(self, min_aspect: float = 2.0, max_aspect: float = 6.5,
                 min_area_frac: float = 0.0008, max_area_frac: float = 0.9):
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac

    def detect(self, image_bgr: np.ndarray, max_detections: int = 5
               ) -> List[Detection]:
        H, W = image_bgr.shape[:2]
        img_area = H * W
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Emphasise vertical character strokes, then find edges.
        grad = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Close with a horizontal kernel scaled to image width so that the gaps
        # *between* characters (which grow with plate size) are reliably bridged
        # into one blob; too-narrow a kernel fragments a plate and yields a crop
        # covering only part of the string.
        kw = max(19, (W // 22) | 1)          # odd, scales with resolution
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (kw // 2, 3)))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Detection] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            area_frac = (w * h) / img_area
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            if not (self.min_area_frac <= area_frac <= self.max_area_frac):
                continue
            # Score by how "plate-like" the region is: fill ratio of edges and a
            # bonus for the ideal ~3.2:1 aspect ratio.
            region = closed[y:y + h, x:x + w]
            fill = float(region.mean()) / 255.0
            aspect_score = 1.0 - min(1.0, abs(aspect - 3.2) / 3.2)
            score = 0.5 * fill + 0.5 * aspect_score
            # Moderate padding so edge characters aren't clipped. The recognizer
            # is trained with matching margin augmentation, so a little extra
            # border is tolerated rather than misread.
            pad_x, pad_y = int(0.04 * w) + 3, int(0.14 * h) + 2
            cx1, cy1 = max(0, x - pad_x), max(0, y - pad_y)
            cx2, cy2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
            crop = image_bgr[cy1:cy2, cx1:cx2].copy()
            candidates.append(Detection(cx1, cy1, cx2, cy2, score, crop))

        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates[:max_detections]


class YoloPlateDetector:
    """Ultralytics-YOLO plate detector (optional, learned backend)."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        try:
            from ultralytics import YOLO  # optional dependency
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'ultralytics' package is required for the YOLO detector. "
                "Install it with: pip install ultralytics"
            ) from exc
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, image_bgr: np.ndarray, max_detections: int = 5
               ) -> List[Detection]:
        results = self.model.predict(image_bgr, conf=self.conf_threshold, verbose=False)
        H, W = image_bgr.shape[:2]
        dets: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                score = float(box.conf[0])
                crop = image_bgr[y1:y2, x1:x2].copy()
                dets.append(Detection(x1, y1, x2, y2, score, crop))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[:max_detections]


def build_detector(cfg) -> Optional[object]:
    """Construct the detector selected in config, with a classical fallback.

    Never raises for a missing YOLO model — it degrades to the classical
    detector so the app always starts.
    """
    d = cfg.detector
    if d.backend == "yolo":
        import os
        weights = cfg.abspath(d.yolo_weights)
        if os.path.exists(weights):
            try:
                return YoloPlateDetector(weights, d.conf_threshold)
            except Exception:
                pass  # fall through to classical
    return ClassicalPlateDetector(d.min_aspect_ratio, d.max_aspect_ratio)
