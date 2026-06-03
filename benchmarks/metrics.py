"""Character/word error-rate metrics via normalized Levenshtein distance."""

from typing import Iterable, Sequence, Tuple


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (chars or word lists)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    """Character error rate: edit distance / len(gt)."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, gt) / len(gt)


def wer(pred: str, gt: str) -> float:
    """Word error rate: token edit distance / number of gt tokens."""
    gt_tokens = gt.split()
    pred_tokens = pred.split()
    if len(gt_tokens) == 0:
        return 0.0 if len(pred_tokens) == 0 else 1.0
    return levenshtein(pred_tokens, gt_tokens) / len(gt_tokens)


def aggregate_cer(pairs: Iterable[Tuple[str, str]]) -> float:
    """Mean per-sample CER over (pred, gt) pairs."""
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(cer(p, g) for p, g in pairs) / len(pairs)


def aggregate_wer(pairs: Iterable[Tuple[str, str]]) -> float:
    """Mean per-sample WER over (pred, gt) pairs."""
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(wer(p, g) for p, g in pairs) / len(pairs)
