"""Offline synthetic license-plate generator.

Why this exists: real plate datasets live on HuggingFace, which isn't always
reachable (corporate proxies, air-gapped CI, this very environment). A synthetic
generator lets the *entire* pipeline — training, evaluation, KPI dashboards —
run end-to-end with zero network access, and it's also a genuinely useful data
augmentation source when real data is scarce.

The generator renders alphanumeric plate strings with real fonts onto
plate-coloured backgrounds, then applies a chain of photometric and geometric
distortions (perspective, rotation, blur, noise, shadows, JPEG artefacts) so the
recognizer learns to be robust rather than memorising a clean font.

Two products are emitted:
  * ``render_plate_crop`` — a tight crop of just the plate (recognizer training).
  * ``render_scene``      — the plate composited onto a larger background with a
                            known bounding box (detector training / demos).
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..charset import ALPHABET

# Candidate fonts, in preference order. Bold monospace reads most like a plate.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

# Common real-world US-style layouts. Each entry is a template where 'A' means a
# random letter and '9' means a random digit. Mixing layouts stops the model
# from over-fitting to a single length or letter/digit arrangement.
_PLATE_TEMPLATES = [
    "AAA9999",   # e.g. ABC1234 (very common)
    "AAA999",
    "999AAA",
    "9AAA999",
    "AA99999",
    "AAA99AA",
    "99AA999",
    "AAAA999",
]

# Plate colour schemes as (background RGB, text RGB). Real plates vary a lot.
_COLOR_SCHEMES = [
    ((245, 245, 245), (25, 25, 30)),     # white plate, dark text
    ((250, 240, 200), (30, 30, 40)),     # pale yellow, dark text
    ((235, 245, 255), (20, 40, 90)),     # light blue tint, navy text
    ((250, 250, 250), (150, 30, 30)),    # white plate, red text
    ((40, 45, 55), (240, 240, 240)),     # dark plate, light text
]


def _available_fonts() -> List[str]:
    fonts = [p for p in _FONT_CANDIDATES if os.path.exists(p)]
    if fonts:
        return fonts
    # Last-ditch fallback: scan for any TTF so the generator never hard-fails.
    for root in ("/usr/share/fonts", "/mnt/skills"):
        found = glob.glob(os.path.join(root, "**", "*.ttf"), recursive=True)
        if found:
            return found[:6]
    return []


_FONTS = _available_fonts()


@dataclass
class PlateSample:
    """One generated example."""
    image: Image.Image           # the rendered image (crop or scene)
    text: str                    # ground-truth plate string
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2) in scene


def random_plate_text(rng: random.Random) -> str:
    """Sample a plate string from a random layout template."""
    template = rng.choice(_PLATE_TEMPLATES)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    out = []
    for ch in template:
        if ch == "A":
            out.append(rng.choice(letters))
        elif ch == "9":
            out.append(rng.choice(digits))
        else:
            out.append(ch)
    return "".join(out)


def _load_font(rng: random.Random, size: int) -> ImageFont.FreeTypeFont:
    if not _FONTS:
        return ImageFont.load_default()
    return ImageFont.truetype(rng.choice(_FONTS), size)


def _draw_clean_plate(text: str, rng: random.Random) -> Image.Image:
    """Render a clean, undistorted plate (colour + text + border)."""
    bg_color, text_color = rng.choice(_COLOR_SCHEMES)
    height = 96
    font_size = int(height * rng.uniform(0.55, 0.68))
    font = _load_font(rng, font_size)

    # Measure text with per-character spacing so glyphs don't overlap.
    spacing = rng.randint(2, 10)
    char_sizes = []
    for ch in text:
        box = font.getbbox(ch)
        char_sizes.append((box[2] - box[0], box[3] - box[1]))
    text_w = sum(w for w, _ in char_sizes) + spacing * (len(text) - 1)
    pad_x = rng.randint(16, 34)
    width = text_w + 2 * pad_x

    plate = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(plate)

    # Rounded-ish border, common on plates.
    border_color = tuple(max(0, c - 60) for c in bg_color)
    draw.rectangle([2, 2, width - 3, height - 3], outline=border_color, width=3)

    # Lay out characters left to right, vertically centred.
    x = pad_x
    for ch, (cw, chh) in zip(text, char_sizes):
        box = font.getbbox(ch)
        y = (height - (box[3] - box[1])) // 2 - box[1]
        draw.text((x - box[0], y), ch, fill=text_color, font=font)
        x += cw + spacing
    return plate


def _apply_perspective(img: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    """Apply a random mild perspective warp to mimic viewing angle."""
    w, h = img.size
    m = strength * min(w, h)

    def jitter() -> float:
        return rng.uniform(-m, m)

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (jitter(), jitter()),
        (w + jitter(), jitter()),
        (w + jitter(), h + jitter()),
        (jitter(), h + jitter()),
    ]
    coeffs = _perspective_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=(127, 127, 127))


def _perspective_coeffs(src, dst):
    """Solve for the 8 PIL perspective-transform coefficients."""
    matrix = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    res = np.linalg.solve(a, b)
    return res.tolist()


def _photometric(img: Image.Image, rng: random.Random) -> Image.Image:
    """Blur, brightness, noise and JPEG-style degradation."""
    arr = np.asarray(img).astype(np.float32)

    # Brightness / contrast jitter.
    arr = arr * rng.uniform(0.7, 1.15) + rng.uniform(-18, 18)

    # Additive Gaussian sensor noise.
    if rng.random() < 0.8:
        arr = arr + np.random.normal(0, rng.uniform(3, 16), arr.shape)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)

    # Optical blur.
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))

    return out


def render_plate_crop(rng: random.Random, augment: bool = True) -> PlateSample:
    """Generate a single augmented plate crop with its label."""
    text = random_plate_text(rng)
    plate = _draw_clean_plate(text, rng)

    if augment:
        if rng.random() < 0.85:
            plate = _apply_perspective(plate, rng, strength=rng.uniform(0.02, 0.09))
        if rng.random() < 0.7:
            angle = rng.uniform(-7, 7)
            plate = plate.rotate(angle, resample=Image.BICUBIC,
                                 expand=False, fillcolor=(127, 127, 127))
        plate = _photometric(plate, rng)

    return PlateSample(image=plate, text=text)


def render_scene(rng: random.Random, size: Tuple[int, int] = (640, 480)) -> PlateSample:
    """Composite a plate onto a textured background and return its bbox.

    Used to demo/train the detection stage without any real photos.
    """
    W, H = size
    # Gradient + noise background so the detector can't cheat on flat colour.
    base = np.zeros((H, W, 3), dtype=np.float32)
    c1 = np.array([rng.randint(30, 120) for _ in range(3)], dtype=np.float32)
    c2 = np.array([rng.randint(60, 180) for _ in range(3)], dtype=np.float32)
    for y in range(H):
        t = y / H
        base[y, :, :] = (1 - t) * c1 + t * c2
    base += np.random.normal(0, 10, base.shape)
    background = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    sample = render_plate_crop(rng, augment=True)
    plate = sample.image
    scale = rng.uniform(0.25, 0.5)
    pw = int(W * scale)
    ph = int(pw * plate.size[1] / plate.size[0])
    plate = plate.resize((pw, ph), Image.BICUBIC)

    x1 = rng.randint(0, max(1, W - pw))
    y1 = rng.randint(0, max(1, H - ph))
    background.paste(plate, (x1, y1))
    return PlateSample(image=background, text=sample.text,
                       bbox=(x1, y1, x1 + pw, y1 + ph))


def generate_dataset(out_dir: str, num_train: int, num_val: int, num_test: int,
                     seed: int = 1234) -> dict:
    """Write train/val/test splits of plate crops plus a labels file.

    Layout::

        out_dir/
          train/000000.png ...   labels.csv (filename,text,split)
          val/...
          test/...

    Returns a small summary dict for logging.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    rows = ["filepath,text,split"]
    counts = {}
    for split, n in (("train", num_train), ("val", num_val), ("test", num_test)):
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(n):
            sample = render_plate_crop(rng, augment=True)
            fname = f"{i:06d}.png"
            sample.image.save(os.path.join(split_dir, fname))
            rows.append(f"{split}/{fname},{sample.text},{split}")
        counts[split] = n

    with open(os.path.join(out_dir, "labels.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    return {"out_dir": out_dir, "counts": counts, "fonts": len(_FONTS),
            "alphabet": ALPHABET}
