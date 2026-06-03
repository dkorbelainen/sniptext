"""Render synthetic screen-text images with known ground truth.

Domain-matched stand-in for SnipText's real input: clean digital text in
light/dark themes, optionally degraded to induce engine disagreement (the
regime where ensemble merging pays off).
"""

import random
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Corpora ---------------------------------------------------------------------
_PROSE = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Software engineers should write code that reads like prose whenever possible.",
    "Performance matters, but clarity and correctness must always come first.",
    "Please review the attached document and send your feedback before Friday.",
    "Machine learning models require careful evaluation on held out data.",
    "The conference will take place in the main hall on the second floor.",
    "Remember to back up your files before installing the latest update.",
    "A good abstraction hides complexity without removing necessary control.",
]
_CODE = [
    "def merge(a, b):\n    return sorted(set(a) | set(b))",
    "for i in range(len(items)):\n    total += items[i].price",
    "class Parser:\n    def __init__(self, path):\n        self.path = path",
    "result = [x * 2 for x in values if x > 0]",
    "if status == 200:\n    return response.json()\nelse:\n    raise Error(status)",
    "import numpy as np\narr = np.zeros((3, 4), dtype=float)",
]
_UI = [
    "File  Edit  View  Help\nNew Project    Ctrl+N\nSave As        Ctrl+S",
    "Settings\nEnable notifications\nDark mode\nAuto save every 5 minutes",
    "Login\nUsername\nPassword\nRemember me    Sign in",
]

_CONTENT = {"prose": _PROSE, "code": _CODE, "ui": _UI}

# Themes: (background, foreground) RGB.
_THEMES = {
    "light": ((250, 250, 250), (20, 20, 20)),
    "dark": ((30, 30, 30), (212, 212, 212)),
}

_FONT_FAMILIES = ["DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono"]


@lru_cache(maxsize=8)
def _font_path(family: str) -> str:
    return (
        subprocess.check_output(["fc-match", "-f", "%{file}", family])
        .decode()
        .strip()
    )


@dataclass
class Sample:
    path: Path
    gt: str
    source: str
    theme: str
    difficulty: str
    content: str


def _render(text: str, theme: str, font_size: int) -> Image.Image:
    bg, fg = _THEMES[theme]
    font = ImageFont.truetype(_font_path(_FONT_FAMILIES[0]), font_size)
    pad = 24
    lines = text.split("\n")
    # Measure block size.
    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)
    widths, height = [], 0
    line_h = font_size + 8
    for ln in lines:
        bbox = d.textbbox((0, 0), ln or " ", font=font)
        widths.append(bbox[2] - bbox[0])
        height += line_h
    w = max(widths) + 2 * pad
    h = height + 2 * pad
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    y = pad
    for ln in lines:
        draw.text((pad, y), ln, font=font, fill=fg)
        y += line_h
    return img


def _degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply moderate noise + blur so engines make differing errors."""
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.1)))
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(
        0, rng.uniform(14, 24), arr.shape
    )
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate(out_dir: Path, n_per_combo: int = 6, seed: int = 42) -> List[Sample]:
    """Render the synthetic corpus into *out_dir*. Returns sample metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    samples: List[Sample] = []
    idx = 0
    for content, texts in _CONTENT.items():
        for theme in _THEMES:
            for difficulty in ("clean", "medium"):
                for _ in range(n_per_combo):
                    text = rng.choice(texts)
                    font_size = rng.choice([22, 26, 30])
                    img = _render(text, theme, font_size)
                    if difficulty == "medium":
                        img = _degrade(img, rng)
                    path = out_dir / f"{content}_{theme}_{difficulty}_{idx:04d}.png"
                    img.save(path)
                    samples.append(
                        Sample(path, text, "synthetic", theme, difficulty, content)
                    )
                    idx += 1
    return samples


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "data" / "synthetic"
    s = generate(root)
    print(f"Generated {len(s)} synthetic samples in {root}")
