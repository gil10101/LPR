"""Recognition metrics and KPIs.

These are the numbers the dashboard presents and the training loop tracks. The
headline recognition metrics for a plate reader are:

  * exact_match_accuracy — fraction of plates read *perfectly* (the KPI that
    matters operationally: a plate is right or it isn't).
  * character_error_rate (CER) — normalised Levenshtein distance; a soft score
    that rewards getting most characters right and is far more informative than
    accuracy alone during training.
  * character_accuracy — 1 - CER, per-character correctness.

We also compute a character confusion matrix (which glyphs get mixed up),
per-length accuracy, and a confidence-vs-correctness table for calibration.
Everything is plain Python/NumPy so it serialises cleanly to JSON/CSV for BI
tools.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    import Levenshtein  # C-accelerated edit distance
    def _edit_distance(a: str, b: str) -> int:
        return Levenshtein.distance(a, b)
except ImportError:  # pragma: no cover - fallback if the wheel is unavailable
    def _edit_distance(a: str, b: str) -> int:
        # Classic DP Levenshtein; fine for short plate strings.
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = cur
        return dp[n]


@dataclass
class RecognitionMetrics:
    """Aggregate recognition metrics plus the raw pieces needed to redraw them."""
    num_samples: int = 0
    exact_match_accuracy: float = 0.0
    character_error_rate: float = 0.0
    character_accuracy: float = 0.0
    mean_edit_distance: float = 0.0
    mean_confidence: float = 0.0
    # accuracy bucketed by ground-truth length -> {length: accuracy}
    accuracy_by_length: Dict[int, float] = field(default_factory=dict)
    # top confused (true_char, pred_char) -> count
    top_confusions: List[Tuple[str, str, int]] = field(default_factory=list)
    # confidence calibration: list of {bucket, mean_conf, accuracy, count}
    calibration: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "exact_match_accuracy": self.exact_match_accuracy,
            "character_error_rate": self.character_error_rate,
            "character_accuracy": self.character_accuracy,
            "mean_edit_distance": self.mean_edit_distance,
            "mean_confidence": self.mean_confidence,
            "accuracy_by_length": {str(k): v for k, v in self.accuracy_by_length.items()},
            "top_confusions": [
                {"true": t, "pred": p, "count": c} for t, p, c in self.top_confusions
            ],
            "calibration": self.calibration,
        }


def _char_confusions(truth: str, pred: str, counter: Counter) -> None:
    """Accumulate substitution confusions via positional alignment.

    For short, roughly-aligned strings a positional zip is a good enough
    approximation of the CTC alignment and keeps the confusion matrix cheap.
    """
    for tc, pc in zip(truth, pred):
        if tc != pc:
            counter[(tc, pc)] += 1


def compute_recognition_metrics(
    predictions: List[str],
    ground_truths: List[str],
    confidences: List[float] | None = None,
    num_confusions: int = 25,
    num_calibration_buckets: int = 10,
) -> RecognitionMetrics:
    """Compute the full metric bundle from parallel prediction/truth lists."""
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    if n == 0:
        return RecognitionMetrics()
    if confidences is None:
        confidences = [0.0] * n

    exact = 0
    total_edits = 0
    total_chars = 0
    edit_distances: List[int] = []
    confusions: Counter = Counter()
    len_correct: Dict[int, int] = defaultdict(int)
    len_total: Dict[int, int] = defaultdict(int)

    for pred, truth in zip(predictions, ground_truths):
        d = _edit_distance(pred, truth)
        edit_distances.append(d)
        total_edits += d
        total_chars += max(len(truth), 1)
        is_exact = pred == truth
        exact += int(is_exact)
        len_total[len(truth)] += 1
        len_correct[len(truth)] += int(is_exact)
        _char_confusions(truth, pred, confusions)

    cer = total_edits / total_chars
    metrics = RecognitionMetrics(
        num_samples=n,
        exact_match_accuracy=exact / n,
        character_error_rate=cer,
        character_accuracy=1.0 - cer,
        mean_edit_distance=float(np.mean(edit_distances)),
        mean_confidence=float(np.mean(confidences)),
        accuracy_by_length={
            length: len_correct[length] / len_total[length]
            for length in sorted(len_total)
        },
        top_confusions=[(t, p, c) for (t, p), c in confusions.most_common(num_confusions)],
        calibration=_calibration_table(predictions, ground_truths, confidences,
                                       num_calibration_buckets),
    )
    return metrics


def _calibration_table(preds, truths, confs, num_buckets: int) -> List[dict]:
    """Bucket predictions by confidence and report accuracy per bucket.

    A well-calibrated model has bucket accuracy tracking bucket confidence.
    """
    buckets: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    for pred, truth, conf in zip(preds, truths, confs):
        b = min(num_buckets - 1, int(conf * num_buckets))
        buckets[b].append((conf, pred == truth))

    table = []
    for b in range(num_buckets):
        items = buckets.get(b, [])
        if not items:
            continue
        mean_conf = float(np.mean([c for c, _ in items]))
        acc = float(np.mean([1.0 if ok else 0.0 for _, ok in items]))
        table.append({
            "bucket": b,
            "range": [b / num_buckets, (b + 1) / num_buckets],
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "count": len(items),
        })
    return table


# ---------------------------------------------------------------------------
# Detection metrics (used when a detector is evaluated against boxed data).
# ---------------------------------------------------------------------------

def iou(box_a, box_b) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_prf(pred_boxes_per_image: List[List[list]],
                  gt_boxes_per_image: List[List[list]],
                  iou_threshold: float = 0.5) -> dict:
    """Precision/recall/F1 and mean IoU of matched boxes at a fixed threshold.

    Greedy one-to-one matching per image: each predicted box claims the highest
    IoU unclaimed ground-truth box above the threshold.
    """
    tp = fp = fn = 0
    matched_ious: List[float] = []
    for preds, gts in zip(pred_boxes_per_image, gt_boxes_per_image):
        used = [False] * len(gts)
        for pb in preds:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gts):
                if used[j]:
                    continue
                v = iou(pb, gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_threshold:
                tp += 1
                used[best_j] = True
                matched_ious.append(best_iou)
            else:
                fp += 1
        fn += used.count(False)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "iou_threshold": iou_threshold,
    }
