"""Thin wrappers running each OCR variant on a single image."""

from pathlib import Path
from typing import Dict

from PIL import Image

from sniptext.config import Config
from sniptext.ensemble import EnsembleOCR, post_process_text
from sniptext.ocr import EasyOCRBackend, TesseractBackend


class EngineRunner:
    """Holds initialised backends so models load once across the corpus."""

    def __init__(self, config: Config, language: str = "eng"):
        self.config = config
        self.language = language
        self.tess = TesseractBackend(config)
        self.easy = EasyOCRBackend(config)
        self.ensemble = EnsembleOCR()

    def run_all(self, image_path: Path) -> Dict[str, str]:
        image = Image.open(image_path).convert("RGB")
        tess_text = self.tess.recognize(image)
        easy_text = self.easy.recognize(image)
        ensemble_text = self.ensemble.combine_results([tess_text, easy_text])
        corrected = post_process_text(
            ensemble_text, language=self.language, enable_correction=True
        )
        return {
            "tesseract": tess_text,
            "easyocr": easy_text,
            "ensemble": ensemble_text,
            "ensemble_corrected": corrected,
        }
