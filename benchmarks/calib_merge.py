"""Close the loop: does calibrating confidence make the confidence-weighted
merge transfer to out-of-domain receipts?

Replays the confidence-weighted merge offline on a held-out image split with
three confidence sources - none (text heuristic), raw engine confidence, and a
per-engine isotonic calibration fit on the train split - and compares mean CER.
No OCR is re-run; the detailed engine outputs are read from merge_inputs.json.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from benchmarks.engines import _flat_word_conf
from benchmarks.metrics import cer, normalize_text
from benchmarks.run_eval import _label_correct
from sniptext.ensemble import EnsembleOCR

_MERGE_INPUTS = Path(__file__).resolve().parent / "merge_inputs.json"
_ENGINES = ("tess", "easy")


def _fit_calibrators(train: List[dict]) -> Dict[str, IsotonicRegression]:
    """Fit a per-engine isotonic conf->correctness map on the train images.

    Per-engine (not pooled) is the point: the merge picks between engines by
    comparing their confidences, so it is invariant to a shared monotone
    rescaling. Only an engine-specific map can re-rank one engine against the
    other - e.g. discount a systematically over-confident engine out of domain.
    """
    isos: Dict[str, IsotonicRegression] = {}
    for e in _ENGINES:
        confs: List[float] = []
        labels: List[int] = []
        for row in train:
            gt_words = normalize_text(row["gt"]).split()
            pairs = _flat_word_conf(row[f"{e}_text"], row[f"{e}_conf"])
            words = [w for w, _ in pairs]
            for (_, c), correct in zip(pairs, _label_correct(words, gt_words)):
                confs.append(c)
                labels.append(correct)
        isos[e] = IsotonicRegression(out_of_bounds="clip").fit(np.array(confs), np.array(labels))
    return isos


def _calibrate(confs, iso: IsotonicRegression):
    """Map per-line word confidences through the calibrator, preserving shape."""
    if not confs:
        return confs
    return [[float(v) for v in iso.predict(np.array(line))] if line else line for line in confs]


def _mean_cer(images: List[dict], confidences) -> float:
    """Mean CER of the merge over images, given a per-image confidence builder."""
    ens = EnsembleOCR()
    total = 0.0
    for row in images:
        texts = [row["tess_text"], row["easy_text"]]
        merged = ens.combine_results(texts, confidences(row))
        total += cer(normalize_text(merged), normalize_text(row["gt"]))
    return total / len(images) if images else float("nan")


def _domain(rows: List[dict], isos: Dict[str, IsotonicRegression]) -> Dict[str, float]:
    raw = lambda r: [r["tess_conf"], r["easy_conf"]]  # noqa: E731
    cal = lambda r: [  # noqa: E731
        _calibrate(r["tess_conf"], isos["tess"]),
        _calibrate(r["easy_conf"], isos["easy"]),
    ]
    return {
        "n_test": len(rows),
        "cer_heuristic": _mean_cer(rows, lambda r: None),
        "cer_rawconf": _mean_cer(rows, raw),
        "cer_calibrated": _mean_cer(rows, cal),
    }


def evaluate() -> Dict[str, Dict[str, float]]:
    data = json.loads(_MERGE_INPUTS.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for src in sorted({r["source"] for r in data}):
        rows = [r for r in data if r["source"] == src]
        if len(rows) < 4:
            continue
        train, test = train_test_split(rows, test_size=0.3, random_state=42)
        out[src] = _domain(test, _fit_calibrators(train))
    if len(out) > 1:
        # Aggregate row: each domain keeps its own calibrator, so pool the
        # per-image means by test population (exact, since CER is a per-image mean).
        keys = ("cer_heuristic", "cer_rawconf", "cer_calibrated")
        domains = list(out.values())
        n = sum(d["n_test"] for d in domains)
        out["all"] = {"n_test": n}
        out["all"].update({k: sum(d[k] * d["n_test"] for d in domains) / n for k in keys})
    return out


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
