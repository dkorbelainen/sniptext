"""OCR engine: Tesseract behind a per-image choice of preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from loguru import logger
from PIL import Image

from .analyzer import ImageAnalyzer
from .config import Config
from .pipelines import LARGE_IMAGE, MAX_ROUTED_PIXELS, Pipeline, recognize
from .router import Router


class OCRError(RuntimeError):
    """Tesseract could not run."""


class OCREngine:
    """Recognises text with the pipeline the router picks for each image."""

    def __init__(self, config: Config):
        self.config = config
        self.router = Router(time_weight=config.router_time_weight)
        self._analyzer = ImageAnalyzer()
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except Exception as e:
            raise OCRError(
                "Tesseract is not available. Install it, for example: "
                "sudo pacman -S tesseract tesseract-data-eng"
            ) from e
        logger.info(f"OCR language: {config.ocr_language}")

    def recognize(self, image: "np.ndarray | Image.Image") -> str:
        """Recognise the text in an image. Raises OCRError when Tesseract cannot run."""
        if isinstance(image, np.ndarray) and image.size == 0:
            return ""
        prepared = self._prepare_image(image)
        if prepared.width == 0 or prepared.height == 0:
            return ""
        text, pipeline = self._route(prepared)
        if not text:
            logger.debug("No text recognized")
            return ""
        logger.info(f"Recognized text: {len(text)} characters (pipeline: {pipeline})")
        return text

    def _run_fixed(self, pipeline: Pipeline, image: Image.Image) -> tuple[str, list[list[float]]]:
        """A pass with nothing to fall back on: its failure is the capture's failure."""
        try:
            return recognize(pipeline, image, self.config.ocr_language)
        except Exception as e:
            raise OCRError(f"Tesseract failed: {e}") from e

    def _run_other(self, pipeline: Pipeline, image: Image.Image) -> str:
        """Text of a non-default pipeline; empty when it fails, so the default takes over."""
        try:
            return recognize(pipeline, image, self.config.ocr_language)[0]
        except Exception as e:
            logger.warning(
                f"Pipeline {pipeline.name} failed ({e}); using {self.router.default.name}"
            )
            return ""

    def _choose(self, image: Image.Image, first_conf: list[list[float]] | None = None) -> Pipeline:
        try:
            return self.router.choose(self._analyzer.extract_features(image), first_conf)
        except Exception as e:
            logger.warning(f"Routing failed ({e}); using {self.router.default.name}")
            return self.router.default

    def _route(self, image: Image.Image) -> tuple[str, str]:
        """Text and the name of the pipeline that produced it."""
        if image.width * image.height > MAX_ROUTED_PIXELS:
            return self._run_fixed(LARGE_IMAGE, image)[0], LARGE_IMAGE.name
        default = self.router.default
        if not (self.config.routing and self.router.available):
            return self._run_fixed(default, image)[0], default.name
        if self.router.policy == "cascade":
            first_text, first_conf = self._run_fixed(default, image)
            chosen = self._choose(image, first_conf)
            text = "" if chosen == default else self._run_other(chosen, image)
            return (text, chosen.name) if text else (first_text, default.name)
        chosen = self._choose(image)
        text = "" if chosen == default else self._run_other(chosen, image)
        return (text, chosen.name) if text else (self._run_fixed(default, image)[0], default.name)

    def _prepare_image(self, image: "np.ndarray | Image.Image") -> Image.Image:
        """An RGB or grayscale copy, transparency over white, no side above max_image_size."""
        pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)
        if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
            rgba = pil_image.convert("RGBA")
            pil_image = Image.new("RGB", rgba.size, "white")
            pil_image.paste(rgba, mask=rgba.getchannel("A"))
        elif pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        limit = self.config.max_image_size
        if max(pil_image.width, pil_image.height) > limit:
            pil_image.thumbnail((limit, limit), Image.LANCZOS)
            logger.debug(f"Image resized to {pil_image.width}x{pil_image.height}")
        return pil_image
