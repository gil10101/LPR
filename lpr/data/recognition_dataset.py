"""PyTorch ``Dataset`` and collate logic for the CRNN recognizer.

The recognizer consumes fixed-size grayscale plate crops and produces variable
length label sequences, so the collate function packs labels into the flat
(concatenated) form that ``torch.nn.CTCLoss`` expects.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..charset import CharsetCodec, normalize_plate_text


def preprocess_crop(image_bgr_or_gray: np.ndarray, height: int, width: int,
                    channels: int) -> np.ndarray:
    """Resize + normalise a raw plate crop into the network's input tensor space.

    Returns a float32 array shaped (C, H, W) with values in [-1, 1]. Kept as a
    free function so training, evaluation and the web app all preprocess
    identically — a common source of train/serve skew otherwise.
    """
    img = image_bgr_or_gray
    if channels == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 127.5 - 1.0
        return img[None, :, :]  # (1, H, W)
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 127.5 - 1.0
        return np.transpose(img, (2, 0, 1))  # (3, H, W)


def _margin_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add/remove border margins to mimic imperfect detector crops.

    This is the key to end-to-end robustness: at inference the detector rarely
    produces a pixel-perfect crop — it adds surrounding scene pixels (loose) or
    clips an edge character (tight). Training only on tight synthetic crops
    leaves the recognizer brittle to that, reading plate borders as extra 'I'/'1'
    characters or dropping edge glyphs. Simulating both here closes the gap.
    """
    h, w = img.shape[:2]
    if rng.random() < 0.75:
        # Loose crop: pad each side with a random solid colour (scene-like) or a
        # dark border, independently per side.
        pads = [int(rng.uniform(0, 0.14) * (w if i < 2 else h)) for i in range(4)]
        left, right, top, bottom = pads
        if rng.random() < 0.5:
            color = [int(rng.uniform(0, 255)) for _ in range(3)]
            border = cv2.BORDER_CONSTANT
            img = cv2.copyMakeBorder(img, top, bottom, left, right, border, value=color)
        else:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    elif rng.random() < 0.4:
        # Slightly tight crop: shave a few edge pixels (mild truncation).
        dx = int(rng.uniform(0, 0.05) * w)
        dy = int(rng.uniform(0, 0.08) * h)
        img = img[dy:h - dy, dx:w - dx]
        if img.size == 0:
            return np.full((h, w, 3), 127, np.uint8)
    return img


def _augment_crop(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light on-the-fly augmentation applied to already-loaded crops.

    Cheap operations only (the heavy geometric distortion happens in the
    synthetic generator); here we add small photometric perturbations plus
    margin augmentation so the recognizer tolerates real detector crops.
    """
    img = _margin_augment(img, rng)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.4:
        alpha = rng.uniform(0.8, 1.2)
        beta = rng.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        noise = np.random.normal(0, rng.uniform(2, 10), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


class PlateRecognitionDataset(Dataset):
    """Maps (image path, plate text) pairs to (tensor, label) training pairs.

    Parameters
    ----------
    samples : list of (absolute_image_path, plate_text)
    codec   : CharsetCodec used to turn text into CTC indices
    augment : apply on-the-fly augmentation (train only)
    """

    def __init__(self, samples: List[Tuple[str, str]], codec: CharsetCodec,
                 img_height: int, img_width: int, channels: int,
                 augment: bool = False, seed: int = 0) -> None:
        self.samples = samples
        self.codec = codec
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, text = self.samples[idx]
        text = normalize_plate_text(text)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Corrupt/missing file: fall back to a blank crop so a bad sample
            # can't crash a whole training run.
            img = np.full((self.img_height, self.img_width, 3), 127, np.uint8)
        if self.augment:
            img = _augment_crop(img, self._rng)
        tensor = preprocess_crop(img, self.img_height, self.img_width, self.channels)
        label = torch.tensor(self.codec.encode(text), dtype=torch.long)
        return torch.from_numpy(tensor), label, text


def ctc_collate(batch):
    """Collate variable-length labels into CTCLoss's flat-target format.

    Returns
    -------
    images        : (B, C, H, W) float tensor
    targets       : (sum_label_lengths,) long tensor, all labels concatenated
    target_lengths: (B,) long tensor, length of each label
    texts         : list[str], ground-truth strings (for eval/logging)
    """
    images, labels, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    if target_lengths.sum() == 0:
        targets = torch.zeros(0, dtype=torch.long)
    else:
        targets = torch.cat([l for l in labels if len(l) > 0])
    return images, targets, target_lengths, list(texts)


def load_samples_from_labels_csv(labels_csv: str, split: Optional[str] = None
                                 ) -> List[Tuple[str, str]]:
    """Read a ``labels.csv`` (filepath,text,split) into absolute-path samples."""
    base = os.path.dirname(os.path.abspath(labels_csv))
    samples: List[Tuple[str, str]] = []
    with open(labels_csv, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            rel, text = parts[0], parts[1]
            row_split = parts[2] if len(parts) > 2 else None
            if split is not None and row_split is not None and row_split != split:
                continue
            samples.append((os.path.join(base, rel), text))
    return samples
