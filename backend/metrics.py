"""Word / character error rate with normalisation suitable for code-mixed text."""
from __future__ import annotations

from .fusion import normalize


def _edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(normalize(reference).replace(" ", "")), list(normalize(hypothesis).replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)
