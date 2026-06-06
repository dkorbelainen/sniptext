"""Thin wrappers running each OCR variant on a single image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from sniptext.config import Config
from sniptext.ensemble import EnsembleOCR, post_process_text
from sniptext.ocr import EasyOCRBackend, TesseractBackend


@dataclass
class RunResult:
    """Per-image OCR outputs plus raw per-word confidences for calibration."""

    texts: Dict[str, str]
    word_conf: Dict[str, List[Tuple[str, float]]]


def _flat_word_conf(text: str, confs: List[List[float]] | None) -> List[Tuple[str, float]]:
    """Flatten detailed (text, per-line word confidences) into (word, conf) pairs.

    recognize_detailed builds each line as " ".join(words) with a parallel
    per-line confidence list, so splitting the text mirrors the conf nesting.
    """
    if not confs:
        return []
    words = [w for line in text.split("\n") for w in line.split()]
    flat = [c for line in confs for c in line]
    return list(zip(words, flat))


class EngineRunner:
    """Holds initialised backends so models load once across the corpus."""

    def __init__(self, config: Config, language: str = "eng"):
        self.config = config
        self.language = language
        self.tess = TesseractBackend(config)
        self.easy = EasyOCRBackend(config)
        self.ensemble = EnsembleOCR()

    def run_all(self, image_path: Path) -> RunResult:
        image = Image.open(image_path).convert("RGB")
        # Canonical path = production fast path (tesseract image_to_string).
        # cer_tesseract, the ensemble accuracy table and the oracle label all
        # derive from these, so they must match what the app actually runs.
        tess_text = self.tess.recognize(image)
        easy_text = self.easy.recognize(image)
        ensemble_text = self.ensemble.combine_results([tess_text, easy_text])
        corrected = post_process_text(ensemble_text, language=self.language, enable_correction=True)

        # Confidence experiment, isolated on the detailed (per-word-confidence)
        # outputs: both merges run over identical detailed inputs so the only
        # difference is heuristic vs confidence-weighted disagreement handling.
        tess_d, tess_conf = self.tess.recognize_detailed(image)
        easy_d, easy_conf = self.easy.recognize_detailed(image)
        ensemble_det_heur = self.ensemble.combine_results([tess_d, easy_d])
        ensemble_det_conf = self.ensemble.combine_results([tess_d, easy_d], [tess_conf, easy_conf])
        return RunResult(
            texts={
                "tesseract": tess_text,
                "easyocr": easy_text,
                "ensemble": ensemble_text,
                "ensemble_corrected": corrected,
                "ensemble_det_heur": ensemble_det_heur,
                "ensemble_det_conf": ensemble_det_conf,
            },
            word_conf={
                "tesseract": _flat_word_conf(tess_d, tess_conf),
                "easyocr": _flat_word_conf(easy_d, easy_conf),
            },
        )
