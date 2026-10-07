"""Run the three router actions on one image, with timing."""

from __future__ import annotations

import time
from dataclasses import dataclass

from PIL import Image

from sniptext.config import Config
from sniptext.ensemble import EnsembleOCR
from sniptext.ocr import EasyOCRBackend, TesseractBackend


@dataclass
class RunResult:
    """Texts per action, per-line word confidences per engine, seconds per engine call."""

    texts: dict
    confs: dict
    times: dict


def _flat_word_conf(text: str, confs: list[list[float]] | None) -> list[tuple[str, float]]:
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

    def __init__(self, config: Config):
        self.tess = TesseractBackend(config)
        self.easy = EasyOCRBackend(config)
        self.ensemble = EnsembleOCR()

    def run_all(self, image: Image.Image) -> RunResult:
        """Same calls, in the same order, as OCREngine makes for each action."""
        start = time.perf_counter()
        tess_plain = self.tess.recognize(image)
        t_plain = time.perf_counter() - start

        start = time.perf_counter()
        tess_text, tess_conf = self.tess.recognize_detailed(image)
        t_tess = time.perf_counter() - start

        start = time.perf_counter()
        easy_text, easy_conf = self.easy.recognize_detailed(image)
        t_easy = time.perf_counter() - start

        merged = self.ensemble.combine_results([tess_text, easy_text], [tess_conf, easy_conf])
        return RunResult(
            texts={
                "tesseract_plain": tess_plain,
                "tesseract": tess_text,
                "easyocr": easy_text,
                "merge": merged,
            },
            confs={"tesseract": tess_conf, "easyocr": easy_conf},
            times={"tesseract_plain": t_plain, "tesseract": t_tess, "easyocr": t_easy},
        )
