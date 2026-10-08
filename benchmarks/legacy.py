"""The frozen results of SnipText 0.4: three actions and the router that chose between them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_LEGACY = Path(__file__).resolve().parent / "legacy_v04.json"


def load_legacy(path=_LEGACY) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def legacy_arrays(legacy: dict, images) -> dict:
    """Per requested image: CER of the three 0.4 actions, the action taken and its seconds."""
    rows = legacy["rows"]
    n = len(images)
    out = {
        "present": np.zeros(n, dtype=bool),
        "cer": np.full((n, 3), np.nan),
        "action": np.full(n, -1, dtype=int),
        "seconds": np.full(n, np.nan),
    }
    for i, image in enumerate(images):
        row = rows.get(str(image))
        if row is None:
            continue
        t_tesseract, t_easyocr = row["time"]
        out["present"][i] = True
        out["cer"][i] = row["cer"]
        out["action"][i] = row["action"]
        out["seconds"][i] = (t_tesseract, t_easyocr, t_tesseract + t_easyocr)[row["action"]]
    return out
