"""CTC decoding helpers shared by evaluation and inference.

Greedy decoding is fast and, for short constrained strings like plates, usually
as good as beam search. We also expose a per-timestep confidence so the UI can
show how sure the model is (mean probability of the emitted characters).
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from ..charset import BLANK_INDEX, CharsetCodec


@torch.no_grad()
def greedy_decode(log_probs: torch.Tensor, codec: CharsetCodec
                  ) -> List[Tuple[str, float]]:
    """Decode a batch of CTC outputs into (text, confidence) pairs.

    Parameters
    ----------
    log_probs : (T, B, num_classes) log-softmax tensor from :class:`CRNN`.
    codec     : maps indices back to characters.

    Confidence is the mean of the max per-timestep probability over the
    timesteps that actually emitted a (non-blank, non-repeat) character. This
    correlates with correctness and is what the dashboard displays.
    """
    probs = log_probs.exp()                       # (T, B, C)
    max_prob, argmax = probs.max(dim=2)           # (T, B) each
    T, B = argmax.shape

    results: List[Tuple[str, float]] = []
    for b in range(B):
        path = argmax[:, b].tolist()
        confs = max_prob[:, b].tolist()
        chars: List[str] = []
        char_confs: List[float] = []
        prev = None
        for t in range(T):
            idx = path[t]
            if idx != prev and idx != BLANK_INDEX:
                chars.append(codec.index_to_char.get(idx, ""))
                char_confs.append(confs[t])
            prev = idx
        text = "".join(chars)
        confidence = float(sum(char_confs) / len(char_confs)) if char_confs else 0.0
        results.append((text, confidence))
    return results
