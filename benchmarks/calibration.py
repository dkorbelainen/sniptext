"""Confidence calibration analysis for per-word OCR confidences.

Reads the (confidence, correct, source) pairs emitted by run_eval and measures
how well raw engine confidence tracks empirical word accuracy (reliability /
expected calibration error), then refits a per-domain isotonic calibrator and
reports the held-out ECE before and after.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

_WORD_CONF = Path(__file__).resolve().parent / "word_conf.json"


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: avg |confidence - accuracy| over equal-width
    confidence bins, weighted by bin population."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1])
    total = len(conf)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _domain(rows) -> Dict[str, float]:
    conf = np.array([r["conf"] for r in rows], dtype=float)
    y = np.array([r["correct"] for r in rows], dtype=int)
    accuracy = float(y.mean())
    mean_conf = float(conf.mean())
    res = {
        "n": len(rows),
        "accuracy": accuracy,
        "mean_conf": mean_conf,
        "ece_raw": _ece(conf, y),
        "ece_cal": float("nan"),
    }
    # Need both classes to fit and to stratify a held-out split.
    if len(np.unique(y)) < 2:
        return res
    c_tr, c_te, y_tr, y_te = train_test_split(conf, y, test_size=0.3, random_state=42, stratify=y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(c_tr, y_tr)
    res["ece_raw"] = _ece(c_te, y_te)
    res["ece_cal"] = _ece(iso.predict(c_te), y_te)
    return res


def calibrate() -> Dict[str, Dict[str, float]]:
    data = json.loads(_WORD_CONF.read_text())
    out: Dict[str, Dict[str, float]] = {}
    sources = sorted({r["source"] for r in data})
    for src in (*sources, "all"):
        rows = data if src == "all" else [r for r in data if r["source"] == src]
        if rows:
            out[src] = _domain(rows)
    return out


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
