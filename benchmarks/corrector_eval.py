"""Does the default-on text correction lower CER? Decided on validation texts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.evaluate import cluster_bootstrap, paired
from benchmarks.metrics import cer, normalize_text
from benchmarks.train_router import delivered, load_arrays
from sniptext.ensemble import post_process_text
from sniptext.router import Router

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "corrector_eval.json"


def measure(data: dict, router, mask: np.ndarray, language: str) -> dict:
    """CER of the text the app delivers, with and without correction."""
    rows = [row for row, keep in zip(data["rows"], mask) if keep]
    raw, corrected, changed = [], [], 0
    for row, column in zip(rows, delivered(data, router, mask)):
        text = row["text"][data["names"][column]]
        fixed = post_process_text(text, language=language, enable_correction=True, aggressive=False)
        gt = normalize_text(row["gt"])
        raw.append(cer(normalize_text(text), gt))
        corrected.append(cer(normalize_text(fixed), gt))
        changed += normalize_text(fixed) != normalize_text(text)
    clusters = data["text_id"][mask]
    return {
        "n": len(rows),
        "cer_raw": cluster_bootstrap(np.minimum(raw, 1.0), clusters),
        "cer_corrected": cluster_bootstrap(np.minimum(corrected, 1.0), clusters),
        "delta": paired(corrected, raw, clusters),
        "changed_share": changed / len(rows),
    }


def decide(delta) -> str:
    """Correction stays only when it lowers CER beyond the interval."""
    return "keep" if delta[2] < 0 else "remove"


def run(
    results=_HERE / "results.json",
    timing=_HERE / "timing.json",
    model=_HERE.parent / "sniptext" / "data" / "router_model.json",
    out=_OUT,
) -> dict:
    data = load_arrays(results, timing)
    router = Router(model)
    val = measure(data, router, data["split"] == "val", data["language"])
    result = {
        "language": data["language"],
        "symspell": importlib.util.find_spec("symspellpy") is not None,
        "val": val,
        "test": measure(data, router, data["split"] == "test", data["language"]),
        "decision": decide(val["delta"]),
    }
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({key: outcome[key] for key in ("symspell", "val", "decision")}, indent=1))
