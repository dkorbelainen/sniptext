"""Policy evaluation: cluster bootstrap, per-policy summaries, weak-label agreement."""

from __future__ import annotations

import numpy as np

from sniptext.router import ACTIONS


def cluster_bootstrap(values, clusters, n_boot: int = 2000, seed: int = 0) -> tuple:
    """Mean of *values* and its 95% interval, resampling whole clusters.

    Images rendered from one text are not independent, so the text is the unit.
    """
    values = np.asarray(values, dtype=float)
    _, index = np.unique(np.asarray(clusters), return_inverse=True)
    n = int(index.max()) + 1
    sums = np.bincount(index, weights=values, minlength=n)
    counts = np.bincount(index, minlength=n)
    draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def realized(matrix, actions) -> np.ndarray:
    """Per row, the entry in the column the policy chose."""
    matrix = np.asarray(matrix)
    return matrix[np.arange(len(matrix)), np.asarray(actions)]


def time_matrix(policy: str, t_tesseract, t_easyocr) -> np.ndarray:
    """Seconds each action takes on each image; mirrors sniptext.router.action_costs."""
    t_tesseract = np.asarray(t_tesseract, dtype=float)
    t_easyocr = np.asarray(t_easyocr, dtype=float)
    both = t_tesseract + t_easyocr
    if policy == "pre_ocr":
        return np.column_stack([t_tesseract, t_easyocr, both])
    if policy == "cascade":
        return np.column_stack([t_tesseract, both, both])
    raise ValueError(f"unknown policy {policy!r}")


def summarize(name: str, cer, actions, times, clusters, reference=None) -> dict:
    """Quality and cost of one policy on one slice.

    *cer* and *times* are (n, 3) per-action matrices, *actions* the chosen column
    per image, *reference* another policy's per-image clipped CER for a paired
    difference.
    """
    cer = np.asarray(cer, dtype=float)
    actions = np.asarray(actions)
    clipped = np.minimum(cer, 1.0)
    chosen = realized(clipped, actions)
    raw = realized(cer, actions)
    out = {
        "policy": name,
        "n": int(len(cer)),
        "cer": cluster_bootstrap(chosen, clusters),
        "cer_unclipped": float(raw.mean()),
        "cer_median": float(np.median(raw)),
        "regret": cluster_bootstrap(chosen - clipped.min(axis=1), clusters),
        "share": {action: float(np.mean(actions == k)) for k, action in enumerate(ACTIONS)},
        "time": float(realized(times, actions).mean()),
    }
    if reference is not None:
        out["delta"] = cluster_bootstrap(chosen - np.asarray(reference, dtype=float), clusters)
    return out


def weak_label_agreement(rows: list) -> dict:
    """Does the app's old quality-score label name the true CER winner?

    The app compared a quality score of the Tesseract text with one of the merged
    text and trained on whichever was higher. Here that label is compared with
    the action that really had the lower CER.
    """
    from sklearn.metrics import cohen_kappa_score

    fast = np.array([r["weak"]["fast_quality"] for r in rows])
    ens = np.array([r["weak"]["ens_quality"] for r in rows])
    weak = (ens > fast).astype(int)
    true = np.array([r["cer"]["merge"] < r["cer"]["tesseract_plain"] for r in rows]).astype(int)
    constant = len(set(weak.tolist())) == 1 or len(set(true.tolist())) == 1
    return {
        "n": len(rows),
        "accuracy": float((weak == true).mean()),
        "kappa": 0.0 if constant else float(cohen_kappa_score(true, weak)),
        "weak_merge_share": float(weak.mean()),
        "true_merge_share": float(true.mean()),
        # The app only recorded a sample when the two scores differed by 0.02.
        "recorded_share": float((np.abs(ens - fast) >= 0.02).mean()),
    }
