"""Policy evaluation: cluster bootstrap, the app's delivery rules, per-policy summaries."""

from __future__ import annotations

import numpy as np

# A bound near zero moves in the fourth decimal at a few thousand resamples.
N_BOOT = 100_000
_CHUNK = 10_000


def cluster_bootstrap(values, clusters, n_boot: int | None = None, seed: int = 0) -> tuple:
    """Mean of *values* and its 95% interval, resampling whole clusters.

    Images rendered from one text are not independent, so the text is the unit.
    """
    values = np.asarray(values, dtype=float)
    _, index = np.unique(np.asarray(clusters), return_inverse=True)
    n = int(index.max()) + 1
    sums = np.bincount(index, weights=values, minlength=n)
    counts = np.bincount(index, minlength=n)
    rng = np.random.default_rng(seed)
    remaining = N_BOOT if n_boot is None else n_boot
    means = []
    while remaining > 0:
        draws = rng.integers(0, n, size=(min(_CHUNK, remaining), n))
        means.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
        remaining -= len(draws)
    low, high = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def realized(matrix, actions) -> np.ndarray:
    """Per row, the entry in the column the policy chose."""
    matrix = np.asarray(matrix)
    return matrix[np.arange(len(matrix)), np.asarray(actions)]


def effective(policy: str, cer, seconds, empty) -> tuple:
    """What choosing each action delivers once the app's rules apply.

    Column 0 is the default pipeline. A non-default action that returns no text
    is replaced by the default's text. A cascade has always run the default
    first; before OCR the default runs only as that replacement.
    """
    cer = np.asarray(cer, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    fallback = np.asarray(empty, dtype=bool).copy()
    fallback[:, 0] = False
    first_cer = np.broadcast_to(cer[:, :1], cer.shape)
    first_seconds = np.broadcast_to(seconds[:, :1], seconds.shape)
    out_cer = np.where(fallback, first_cer, cer)
    out_seconds = seconds.copy()
    if policy == "cascade":
        out_seconds[:, 1:] += seconds[:, :1]
    elif policy == "pre_ocr":
        out_seconds = np.where(fallback, seconds + first_seconds, seconds)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return out_cer, out_seconds


def summarize(name, cer, seconds, clusters, oracle=None, reference=None, share=None) -> dict:
    """Quality and cost of one policy on one slice, from its per-image CER and seconds.

    *oracle* and *reference* are per-image clipped CER of the oracle and of the
    policy to compare with; *share* is how often each action was taken.
    """
    cer = np.asarray(cer, dtype=float)
    clipped = np.minimum(cer, 1.0)
    out = {
        "policy": name,
        "n": int(len(cer)),
        "cer": cluster_bootstrap(clipped, clusters),
        "cer_unclipped": float(cer.mean()),
        "cer_median": float(np.median(cer)),
        "time": float(np.mean(seconds)),
    }
    if oracle is not None:
        out["regret"] = cluster_bootstrap(clipped - np.asarray(oracle, dtype=float), clusters)
    if reference is not None:
        out["delta"] = cluster_bootstrap(clipped - np.asarray(reference, dtype=float), clusters)
    if share is not None:
        out["share"] = share
    return out


def paired(a, b, clusters) -> tuple:
    """Mean and interval of the per-image difference of two clipped CER vectors."""
    a = np.minimum(np.asarray(a, dtype=float), 1.0)
    b = np.minimum(np.asarray(b, dtype=float), 1.0)
    return cluster_bootstrap(a - b, clusters)
