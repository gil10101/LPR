"""Fast, dependency-light tests for the core LPR components.

Run with:  python -m pytest tests/  (or) python tests/test_core.py

These cover the pure logic that's easy to get subtly wrong — the CTC codec,
greedy decoding, the metric math, and the CRNN's tensor shapes — without needing
a trained model or network access.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lpr.charset import DEFAULT_CODEC, normalize_plate_text
from lpr.eval.metrics import (compute_recognition_metrics, detection_prf, iou)
from lpr.models.crnn import CRNN
from lpr.utils.ctc import greedy_decode


def test_codec_roundtrip():
    codec = DEFAULT_CODEC
    for text in ["ABC1234", "9XY88ZZ", "000AAA"]:
        idx = codec.encode(text)
        assert codec.decode_indices(idx) == text
    # blank index 0 must never be produced by encode
    assert 0 not in codec.encode("ABCDEFG")


def test_normalize_plate_text():
    assert normalize_plate_text("ab-12 34") == "AB1234"
    assert normalize_plate_text("  xyz  ") == "XYZ"


def test_ctc_greedy_collapse():
    codec = DEFAULT_CODEC
    a = codec.char_to_index["A"]
    b = codec.char_to_index["B"]
    # raw path: A A blank A B B -> "AAB"  (merge repeats, drop blanks)
    assert codec.ctc_greedy_decode([a, a, 0, a, b, b]) == "AAB"
    assert codec.ctc_greedy_decode([0, 0, 0]) == ""


def test_recognition_metrics():
    preds = ["ABC1234", "XYZ0000", "AAA111"]
    truth = ["ABC1234", "XYZ0009", "BBB111"]
    m = compute_recognition_metrics(preds, truth)
    assert m.num_samples == 3
    # one perfect match out of three
    assert abs(m.exact_match_accuracy - 1 / 3) < 1e-9
    # 4 substitutions over 20 ground-truth chars -> CER = 0.20 exactly
    assert abs(m.character_error_rate - 0.2) < 1e-9
    assert abs(m.character_accuracy - (1 - m.character_error_rate)) < 1e-9


def test_iou_and_detection_prf():
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    prf = detection_prf([[[0, 0, 10, 10]]], [[[0, 0, 10, 10]]])
    assert prf["precision"] == 1.0 and prf["recall"] == 1.0


def test_crnn_shapes_and_decode():
    codec = DEFAULT_CODEC
    model = CRNN(codec.num_classes, in_channels=1, rnn_hidden=64, rnn_layers=1,
                 img_height=32)
    model.eval()
    x = torch.randn(4, 1, 32, 128)
    with torch.no_grad():
        out = model(x)                    # (T, B, num_classes)
    assert out.shape[1] == 4
    assert out.shape[2] == codec.num_classes
    decoded = greedy_decode(out, codec)
    assert len(decoded) == 4
    for text, conf in decoded:
        assert isinstance(text, str)
        assert 0.0 <= conf <= 1.0


def test_synthetic_generator_smoke():
    from lpr.data.synthetic import render_plate_crop
    rng = random.Random(0)
    sample = render_plate_crop(rng)
    assert sample.text and normalize_plate_text(sample.text) == sample.text
    assert np.asarray(sample.image).ndim == 3


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
