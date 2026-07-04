"""Character set and CTC codec for license-plate recognition.

The recognizer is a sequence model trained from scratch with a CTC loss. CTC
needs a dedicated *blank* symbol that is never part of a real label, so index 0
is reserved for it and the real characters occupy indices 1..N.

This module is the single source of truth for how strings map to integer label
sequences and back. Keep it dependency-free (only the standard library) so it
can be imported by data tooling, training, evaluation and the web app alike.
"""
from __future__ import annotations

from typing import List, Sequence

# Alphabet used on the plates we recognise: digits then uppercase letters.
# Order is arbitrary but must stay stable once a model is trained, because the
# integer indices are baked into the network's output layer.
ALPHABET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Index 0 is the CTC blank. Real characters start at 1.
BLANK_INDEX: int = 0


class CharsetCodec:
    """Bidirectional mapping between plate strings and CTC label indices."""

    def __init__(self, alphabet: str = ALPHABET) -> None:
        self.alphabet = alphabet
        # char -> index (offset by 1 to leave room for the blank at 0)
        self.char_to_index = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.index_to_char = {i + 1: ch for i, ch in enumerate(alphabet)}

    @property
    def num_classes(self) -> int:
        """Total logits the network emits: alphabet size + 1 blank."""
        return len(self.alphabet) + 1

    def encode(self, text: str) -> List[int]:
        """Turn a plate string into a list of label indices.

        Characters outside the alphabet are silently dropped; callers should
        normalise (upper-case, strip separators) before encoding.
        """
        return [self.char_to_index[ch] for ch in text if ch in self.char_to_index]

    def decode_indices(self, indices: Sequence[int]) -> str:
        """Map a raw index sequence to text (no CTC collapsing)."""
        return "".join(self.index_to_char.get(int(i), "") for i in indices)

    def ctc_greedy_decode(self, indices: Sequence[int]) -> str:
        """Collapse a raw per-timestep argmax path into a string.

        CTC decoding rule: merge runs of the same index, then drop blanks.
        """
        out: List[str] = []
        prev = None
        for idx in indices:
            idx = int(idx)
            if idx != prev and idx != BLANK_INDEX:
                out.append(self.index_to_char.get(idx, ""))
            prev = idx
        return "".join(out)


def normalize_plate_text(text: str) -> str:
    """Canonicalise a plate label: upper-case and keep only alphabet chars.

    Real datasets contain spaces, dashes and lower-case letters; we recognise a
    compact alphanumeric string, so strip everything else.
    """
    return "".join(ch for ch in text.upper() if ch in ALPHABET)


# A module-level default instance is convenient and safe because the codec is
# stateless once constructed.
DEFAULT_CODEC = CharsetCodec()
