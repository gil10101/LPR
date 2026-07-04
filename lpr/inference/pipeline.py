"""End-to-end inference: photo in, plate string(s) out.

This is what the web app and batch evaluation both call. It wires the detection
stage to the recognizer and returns structured results (box, text, confidence,
timings) that the UI can render directly.

Loading is lazy and cached: the CRNN weights are read once on first use. If no
trained weights exist yet, ``PlateRecognizer`` raises a clear error telling the
user to train first.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch

from ..charset import CharsetCodec
from ..config import Config, load_config
from ..data.recognition_dataset import preprocess_crop
from ..models.crnn import CRNN
from ..models.detector import Detection, build_detector
from ..utils.ctc import greedy_decode


@dataclass
class PlateResult:
    """One recognised plate."""
    text: str
    confidence: float
    box: List[int]           # [x1, y1, x2, y2] in the original image
    detection_score: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "box": [int(v) for v in self.box],
            "detection_score": round(self.detection_score, 4),
        }


@dataclass
class PipelineOutput:
    plates: List[PlateResult] = field(default_factory=list)
    detect_ms: float = 0.0
    recognize_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "plates": [p.to_dict() for p in self.plates],
            "timing_ms": {
                "detect": round(self.detect_ms, 2),
                "recognize": round(self.recognize_ms, 2),
                "total": round(self.total_ms, 2),
            },
        }


class PlateRecognizer:
    """Loads the trained CRNN and decodes plate crops to strings."""

    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location=self.device)
        mc = ckpt["config"]
        self.codec = CharsetCodec(ckpt.get("alphabet"))
        self.img_height = mc["img_height"]
        self.img_width = mc["img_width"]
        self.channels = mc["channels"]
        self.model = CRNN(
            num_classes=mc["num_classes"], in_channels=mc["channels"],
            rnn_hidden=mc["rnn_hidden"], rnn_layers=mc["rnn_layers"],
            img_height=mc["img_height"], dropout=mc.get("dropout", 0.1),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.val_accuracy = ckpt.get("val_accuracy")

    @torch.no_grad()
    def read(self, crop_bgr: np.ndarray) -> tuple:
        """Recognise a single plate crop -> (text, confidence)."""
        tensor = preprocess_crop(crop_bgr, self.img_height, self.img_width,
                                 self.channels)
        batch = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        log_probs = self.model(batch)
        (text, conf), = greedy_decode(log_probs, self.codec)
        return text, conf

    @torch.no_grad()
    def read_batch(self, crops: List[np.ndarray]) -> List[tuple]:
        if not crops:
            return []
        tensors = [preprocess_crop(c, self.img_height, self.img_width,
                                   self.channels) for c in crops]
        batch = torch.from_numpy(np.stack(tensors)).to(self.device)
        log_probs = self.model(batch)
        return greedy_decode(log_probs, self.codec)


class LPRPipeline:
    """Detection + recognition, cached for repeated calls (e.g. a web server)."""

    def __init__(self, config_path: Optional[str] = None):
        self.cfg: Config = load_config(config_path)
        self.detector = build_detector(self.cfg)
        self._recognizer: Optional[PlateRecognizer] = None

    @property
    def recognizer(self) -> PlateRecognizer:
        if self._recognizer is None:
            import os
            weights = self.cfg.abspath(self.cfg.recognizer.weights_path)
            if not os.path.exists(weights):
                raise FileNotFoundError(
                    f"No trained recognizer at {weights}. Train one first:\n"
                    f"  python scripts/train_recognizer.py"
                )
            self._recognizer = PlateRecognizer(weights)
        return self._recognizer

    def run(self, image_bgr: np.ndarray, max_plates: int = 5) -> PipelineOutput:
        """Detect plates, recognise each, return structured results."""
        t_start = time.time()

        t0 = time.time()
        detections: List[Detection] = self.detector.detect(image_bgr, max_plates)
        detect_ms = (time.time() - t0) * 1000

        # If detection found nothing, assume the upload is already a plate crop.
        if not detections:
            H, W = image_bgr.shape[:2]
            detections = [Detection(0, 0, W, H, 0.0, image_bgr.copy())]

        t1 = time.time()
        crops = [d.crop for d in detections]
        decoded = self.recognizer.read_batch(crops)
        recognize_ms = (time.time() - t1) * 1000

        plates: List[PlateResult] = []
        for det, (text, conf) in zip(detections, decoded):
            if not text:
                continue
            plates.append(PlateResult(text=text, confidence=conf, box=det.box,
                                      detection_score=det.score))
        # Best (most confident) plate first.
        plates.sort(key=lambda p: p.confidence, reverse=True)

        total_ms = (time.time() - t_start) * 1000
        return PipelineOutput(plates=plates, detect_ms=detect_ms,
                              recognize_ms=recognize_ms, total_ms=total_ms)

    def annotate(self, image_bgr: np.ndarray, output: PipelineOutput
                 ) -> np.ndarray:
        """Draw boxes + predicted text onto a copy of the image for display."""
        img = image_bgr.copy()
        for p in output.plates:
            x1, y1, x2, y2 = p.box
            color = (0, 200, 0) if p.confidence >= 0.7 else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{p.text} {p.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ly = max(0, y1 - th - 8)
            cv2.rectangle(img, (x1, ly), (x1 + tw + 8, ly + th + 8), color, -1)
            cv2.putText(img, label, (x1 + 4, ly + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        return img
