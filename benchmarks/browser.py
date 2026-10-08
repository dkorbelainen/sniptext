"""Render held-out texts in a real browser: an evaluation-only slice.

The synthetic corpus is drawn with Pillow. These pages go through Chrome's text
stack instead (hinting, subpixel positioning, real layout), at several device
scale factors. Ground truth is the text each page was built from.
"""

from __future__ import annotations

import dataclasses
import html
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageChops

from benchmarks.corpus import TextItem, load_items
from benchmarks.synthetic import Sample, assign_splits

_OUT = Path(__file__).resolve().parent / "data" / "browser"
_BROWSERS = ("google-chrome-stable", "chromium", "chromium-browser")
THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1f2328", "panel": "#f6f8fa", "border": "#d0d7de",
              "kw": "#cf222e", "str": "#0a3069", "com": "#6e7781"},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "panel": "#161b22", "border": "#30363d",
             "kw": "#ff7b72", "str": "#a5d6ff", "com": "#8b949e"},
}  # fmt: skip
SCALES = (1.0, 1.5, 2.0)
FONT_SIZES = tuple(range(11, 17))
PROSE_FONTS = ("sans-serif", "serif")
UI_STYLES = ("buttons", "menu", "table")
N_TEXTS = 100
RENDERS_PER_TEXT = 2
_WIDTH = 760
_HEIGHT = 1200
_MARGIN = 24
_PADDING = 12
_KEYWORDS = (
    "def|class|return|if|elif|else|for|while|import|from|in|not|and|or|None|True|False|"
    "try|except|with|as|raise|yield|lambda"
)
_TOKEN = re.compile(rf"(#.*$)|(\"[^\"\n]*\"|'[^'\n]*')|\b({_KEYWORDS})\b", re.MULTILINE)


def _code(text: str) -> str:
    out, position = [], 0
    for match in _TOKEN.finditer(text):
        kind = "com" if match.group(1) else "str" if match.group(2) else "kw"
        out.append(html.escape(text[position : match.start()]))
        out.append(f'<span class="{kind}">{html.escape(match.group(0))}</span>')
        position = match.end()
    out.append(html.escape(text[position:]))
    return "<pre>" + "".join(out) + "</pre>"


def _ui(text: str, style: str) -> str:
    lines = [html.escape(line) for line in text.split("\n")]
    if style == "buttons":
        return "".join(f'<div><span class="button">{line}</span></div>' for line in lines)
    if style == "menu":
        return "<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>"
    if style == "table":
        return "<table>" + "".join(f"<tr><td>{line}</td></tr>" for line in lines) + "</table>"
    raise ValueError(f"unknown UI style {style!r}")


def page_html(item: TextItem, theme: str, font_px: int, family: str, ui_style: str) -> str:
    """One page showing the item's text and nothing else."""
    c = THEMES[theme]
    if item.content == "code":
        body = _code(item.text)
    elif item.content == "ui":
        body = _ui(item.text, ui_style)
    else:
        body = "".join(f"<p>{html.escape(line)}</p>" for line in item.text.split("\n"))
    border = f"1px solid {c['border']}"
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        f"html,body{{margin:0;background:{c['bg']}}}"
        f"body{{padding:{_MARGIN}px;width:{_WIDTH}px;color:{c['fg']};"
        f"font:{font_px}px/1.5 {family}}}"
        "p{margin:0 0 0.6em}"
        f"pre{{margin:0;padding:12px;font:{font_px}px/1.45 monospace;white-space:pre-wrap;"
        f"background:{c['panel']};border:{border};border-radius:6px}}"
        f".kw{{color:{c['kw']}}}.str{{color:{c['str']}}}.com{{color:{c['com']}}}"
        f".button{{display:inline-block;margin:4px 0;padding:6px 14px;background:{c['panel']};"
        f"border:{border};border-radius:6px}}"
        f"ul{{list-style:none;margin:0;padding:6px 0;width:260px;background:{c['panel']};"
        f"border:{border};border-radius:8px}}li{{padding:6px 16px}}"
        f"table{{border-collapse:collapse}}td{{padding:6px 12px;border:{border}}}"
        f"</style></head><body>{body}</body></html>"
    )


def trim(image: Image.Image, padding: int) -> Image.Image:
    """Crop to everything that differs from the page background, plus padding."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
    box = ImageChops.difference(rgb, background).getbbox()
    if box is None:
        return rgb
    left, top, right, bottom = box
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def find_chrome() -> str:
    for name in _BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("Chrome or Chromium is needed to render the browser slice")


def chrome_render(page: str, png: Path, scale: float) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "page.html"
        source.write_text(page, encoding="utf-8")
        subprocess.run(
            [
                find_chrome(),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                f"--user-data-dir={Path(tmp) / 'profile'}",
                f"--force-device-scale-factor={scale:g}",
                f"--window-size={_WIDTH + 2 * _MARGIN},{_HEIGHT}",
                f"--screenshot={png}",
                source.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )


def generate(out_dir, items: list[TextItem], seed: int = 42, render=chrome_render) -> list[Sample]:
    """Render test-split texts into *out_dir* and return their metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(items, seed)
    test = sorted((i for i in items if splits[i.text_id] == "test"), key=lambda i: i.text_id)
    chosen = random.Random(f"{seed}:browser").sample(test, min(N_TEXTS, len(test)))
    samples: list[Sample] = []
    for item in chosen:
        rng = random.Random(f"{seed}:browser:{item.text_id}")
        for number in range(RENDERS_PER_TEXT):
            theme = rng.choice(sorted(THEMES))
            scale = rng.choice(SCALES)
            size = rng.choice(FONT_SIZES)
            family = "monospace" if item.content == "code" else rng.choice(PROSE_FONTS)
            style = rng.choice(UI_STYLES)
            path = out_dir / f"{item.text_id}_b{number}.png"
            render(page_html(item, theme, size, family, style), path, scale)
            with Image.open(path) as shot:
                trimmed = trim(shot, round(_PADDING * scale))
            trimmed.save(path)
            samples.append(
                Sample(
                    path=path,
                    gt=item.text,
                    source="browser",
                    split="browser",
                    text_id=item.text_id,
                    lang=item.lang,
                    content=item.content,
                    font=family,
                    font_size=size,
                    theme=theme,
                    degradation="none",
                    scale=scale,
                )
            )
    return samples


def _environment() -> dict:
    def run(*command: str) -> str:
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "chrome": run(find_chrome(), "--version"),
        "fonts": {family: run("fc-match", family) for family in (*PROSE_FONTS, "monospace")},
    }


def main() -> None:
    samples = generate(_OUT, load_items(42))
    manifest = [{**dataclasses.asdict(s), "path": str(s.path)} for s in samples]
    (_OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (_OUT / "environment.json").write_text(json.dumps(_environment(), indent=1))
    print(f"Rendered {len(samples)} pages into {_OUT}")


if __name__ == "__main__":
    main()
