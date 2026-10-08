"""Render screen-text images with known ground truth.

Each text is rendered several times with different fonts, sizes, colour schemes
and degradations. Splits are assigned per text, so a text never appears on both
sides of a split. Two font families are held out for an unseen-font slice.
"""

from __future__ import annotations

import io
import random
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from benchmarks.corpus import TextItem

SEEN_FONTS = (
    "DejaVu Sans",
    "DejaVu Sans Mono",
    "Liberation Sans",
    "Liberation Mono",
    "Liberation Serif",
    "Noto Sans",
    "Noto Serif",
    "JetBrains Mono",
)
UNSEEN_FONTS = ("Open Sans", "Noto Sans Mono")

# (background, foreground) RGB.
THEMES = {
    "light": ((250, 250, 250), (20, 20, 20)),
    "dark": ((30, 30, 30), (212, 212, 212)),
    "solarized_light": ((253, 246, 227), (101, 123, 131)),
    "solarized_dark": ((0, 43, 54), (131, 148, 150)),
    "terminal": ((12, 12, 12), (51, 255, 51)),
    "paper": ((244, 236, 216), (59, 47, 47)),
}

RENDERS_PER_TEXT = 4
_SPLIT_SHARES = (("train", 0.6), ("val", 0.2), ("test", 0.2))
CONFIRM_SEED = 1042
CONFIRM_TEXTS = 150
_RUN = 6


@dataclass
class Sample:
    path: Path
    gt: str
    source: str
    split: str
    text_id: str
    lang: str
    content: str
    font: str
    font_size: int
    theme: str
    degradation: str
    scale: float = 1.0


@lru_cache(maxsize=None)
def _font_file(family: str) -> str:
    """Path of the font file for *family*; an error if fontconfig would substitute another."""
    out = subprocess.check_output(["fc-match", "-f", "%{family}|%{file}", family]).decode()
    matched, _, path = out.partition("|")
    if family.lower() not in [name.strip().lower() for name in matched.split(",")]:
        raise RuntimeError(f"font family not installed: {family} (fontconfig offers {matched})")
    return path.strip()


def _load_font(family: str, size: int):
    return ImageFont.truetype(_font_file(family), size)


def render(text: str, family: str, size: int, theme: str) -> Image.Image:
    """Draw *text* line by line on a solid background."""
    background, foreground = THEMES[theme]
    font = _load_font(family, size)
    lines = text.split("\n")
    line_height = size + max(4, round(size * 0.35))
    pad = 16
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    widths = [probe.textbbox((0, 0), line or " ", font=font)[2] for line in lines]
    image = Image.new(
        "RGB", (max(widths) + 2 * pad, line_height * len(lines) + 2 * pad), background
    )
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(lines):
        draw.text((pad, pad + row * line_height), line, font=font, fill=foreground)
    return image


def _blur(image: Image.Image, rng: random.Random) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.6)))


def _noise(image: Image.Image, rng: random.Random) -> Image.Image:
    pixels = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
        0.0, rng.uniform(6.0, 34.0), pixels.shape
    )
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


def _jpeg(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=rng.randint(8, 45))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _rescale(image: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.4, 0.8)
    width, height = image.size
    small = image.resize(
        (max(8, int(width * factor)), max(8, int(height * factor))), Image.BILINEAR
    )
    return small.resize((width, height), Image.BILINEAR)


def _lowcontrast(image: Image.Image, rng: random.Random) -> Image.Image:
    mean = tuple(int(channel) for channel in ImageStat.Stat(image).mean)
    return Image.blend(image, Image.new("RGB", image.size, mean), rng.uniform(0.45, 0.8))


DEGRADATIONS = {
    "blur": _blur,
    "noise": _noise,
    "jpeg": _jpeg,
    "rescale": _rescale,
    "lowcontrast": _lowcontrast,
}


def _pick_degradations(rng: random.Random) -> tuple:
    """None for a quarter of images, one for half, two for a quarter."""
    roll = rng.random()
    count = 0 if roll < 0.25 else 1 if roll < 0.75 else 2
    return tuple(rng.sample(sorted(DEGRADATIONS), count))


def assign_splits(items: list[TextItem], seed: int = 42) -> dict[str, str]:
    """Text id to split, 60/20/20, stratified by language and content type."""
    groups: dict[tuple, list[str]] = {}
    for item in items:
        groups.setdefault((item.lang, item.content), []).append(item.text_id)
    splits: dict[str, str] = {}
    for key in sorted(groups):
        ids = sorted(groups[key])
        random.Random(f"{seed}:{key}").shuffle(ids)
        start = 0
        for position, (name, share) in enumerate(_SPLIT_SHARES):
            last = position == len(_SPLIT_SHARES) - 1
            end = len(ids) if last else start + round(len(ids) * share)
            for text_id in ids[start:end]:
                splits[text_id] = name
            start = end
    return splits


def _render_item(out_dir: Path, item: TextItem, plans: list, rng: random.Random) -> list[Sample]:
    """One image per (font family, split) plan, drawing sizes and degradations from *rng*."""
    samples = []
    for number, (family, split) in enumerate(plans):
        size = rng.randint(11, 30)
        theme = rng.choice(sorted(THEMES))
        names = _pick_degradations(rng)
        image = render(item.text, family, size, theme)
        for name in names:
            image = DEGRADATIONS[name](image, rng)
        path = out_dir / f"{item.text_id}_{number}.png"
        image.save(path)
        samples.append(
            Sample(
                path=path,
                gt=item.text,
                source="synthetic",
                split=split,
                text_id=item.text_id,
                lang=item.lang,
                content=item.content,
                font=family,
                font_size=size,
                theme=theme,
                degradation="+".join(names) or "none",
            )
        )
    return samples


def generate(out_dir: Path, items: list[TextItem], seed: int = 42) -> list[Sample]:
    """Render the corpus into *out_dir* and return its metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(items, seed)
    samples: list[Sample] = []
    for item in items:
        # A per-text generator keeps one text's renders stable when the pool changes.
        rng = random.Random(f"{seed}:{item.text_id}")
        split = splits[item.text_id]
        plans = [(rng.choice(SEEN_FONTS), split) for _ in range(RENDERS_PER_TEXT)]
        if split == "test":
            plans += [(family, "unseen_font") for family in UNSEEN_FONTS]
        samples += _render_item(out_dir, item, plans, rng)
    return samples


def generate_slice(out_dir: Path, items: list[TextItem], split: str, seed: int) -> list[Sample]:
    """Render every text like a held-out one, all under a single split name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for item in items:
        rng = random.Random(f"{seed}:{item.text_id}")
        plans = [(rng.choice(SEEN_FONTS), split) for _ in range(RENDERS_PER_TEXT)]
        samples += _render_item(out_dir, item, plans, rng)
    return samples


def _word_runs(text: str) -> set[str]:
    words = text.split()
    return {" ".join(words[k : k + _RUN]) for k in range(max(1, len(words) - _RUN + 1))}


def fresh_items(
    candidates: list[TextItem], used: list[TextItem], n: int, seed: int
) -> list[TextItem]:
    """*n* candidates sharing no six-word run with a used text, in the used pool's proportions."""
    taken = {item.text_id for item in used}
    runs: set[str] = set()
    shares: dict[tuple, int] = {}
    for item in used:
        runs |= _word_runs(item.text)
        key = (item.lang, item.content)
        shares[key] = shares.get(key, 0) + 1
    groups: dict[tuple, list[TextItem]] = {}
    for item in candidates:
        if item.text_id not in taken and not (_word_runs(item.text) & runs):
            groups.setdefault((item.lang, item.content), []).append(item)
    out: list[TextItem] = []
    for key in sorted(shares):
        want = round(n * shares[key] / len(used))
        have = sorted(groups.get(key, []), key=lambda item: item.text_id)
        if len(have) < want:
            raise ValueError(f"only {len(have)} fresh texts for {key}, {want} needed")
        out += random.Random(f"{seed}:fresh:{key}").sample(have, want)
    return out


if __name__ == "__main__":
    from benchmarks.corpus import load_items

    root = Path(__file__).resolve().parent / "data" / "synthetic"
    made = generate(root, load_items())
    by_split: dict[str, int] = {}
    for sample in made:
        by_split[sample.split] = by_split.get(sample.split, 0) + 1
    print(len(made), by_split)
