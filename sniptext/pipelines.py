"""Tesseract pipelines: an image transform plus a page-segmentation rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image, ImageFilter, ImageOps, ImageStat

from .analyzer import ImageAnalyzer

_ANALYZER = ImageAnalyzer()
_DARK_BELOW = 100
# The router is fitted on images far smaller than this; a larger one is not routed.
MAX_ROUTED_PIXELS = 2_000_000


def _enhance(image: Image.Image) -> Image.Image:
    return _ANALYZER.enhance_for_ocr(image)


def _light(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    if sum(ImageStat.Stat(rgb).mean) / 3 < _DARK_BELOW:
        return ImageOps.invert(rgb)
    return rgb


def _up2(image: Image.Image) -> Image.Image:
    return image.resize((image.width * 2, image.height * 2), Image.LANCZOS)


def _median3(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.MedianFilter(3))


def _gauss1(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(1.0))


STEPS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "enhance": _enhance,
    "light": _light,
    "up2": _up2,
    "median3": _median3,
    "gauss1": _gauss1,
}
PSM_RULES = ("auto", "6", "11")


@dataclass(frozen=True)
class Pipeline:
    name: str
    steps: tuple[str, ...]
    psm: str

    def __post_init__(self) -> None:
        unknown = [step for step in self.steps if step not in STEPS]
        if unknown:
            raise ValueError(f"pipeline {self.name!r}: unknown steps {unknown}")
        if self.psm not in PSM_RULES:
            raise ValueError(f"pipeline {self.name!r}: unknown PSM rule {self.psm!r}")

    def prepare(self, image: Image.Image) -> Image.Image:
        for step in self.steps:
            image = STEPS[step](image)
        return image

    def psm_mode(self, prepared: Image.Image) -> int:
        """Page segmentation mode for an already prepared image."""
        if self.psm == "auto":
            return _ANALYZER.suggest_psm_mode(prepared)
        return int(self.psm)

    def to_dict(self) -> dict:
        return {"name": self.name, "steps": list(self.steps), "psm": self.psm}

    @classmethod
    def from_dict(cls, data: dict) -> "Pipeline":
        return cls(str(data["name"]), tuple(data["steps"]), str(data["psm"]))


V04 = Pipeline("enhance_auto", ("enhance",), "auto")
DEFAULT = V04


def recognize(pipeline: Pipeline, image: Image.Image, lang: str) -> tuple[str, list[list[float]]]:
    """Text and per-line word confidences in [0, 1] from one Tesseract pass."""
    import pytesseract

    prepared = pipeline.prepare(image)
    data = pytesseract.image_to_data(
        prepared,
        lang=lang,
        config=f"--oem 1 --psm {pipeline.psm_mode(prepared)}",
        output_type=pytesseract.Output.DICT,
    )
    grouped: dict[tuple, list[tuple[int, str, float]]] = {}
    for k, raw in enumerate(data["text"]):
        word = raw.strip()
        conf = float(data["conf"][k])
        if not word or conf < 0:
            continue
        key = (data["block_num"][k], data["par_num"][k], data["line_num"][k])
        grouped.setdefault(key, []).append((data["word_num"][k], word, conf / 100.0))
    lines: list[str] = []
    confs: list[list[float]] = []
    for words in grouped.values():
        words.sort(key=lambda entry: entry[0])
        lines.append(" ".join(word for _, word, _ in words))
        confs.append([conf for _, _, conf in words])
    return "\n".join(lines), confs
