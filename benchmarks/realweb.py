"""Text elements of real web pages: the element OCR task of VisualWebBench.

Each row is a page screenshot with one element framed in red, the element's box
and its text. The crop of that box, without the frame, is one sample. Run this
file once before run_eval.py; a missing manifest only skips the slice.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from benchmarks.synthetic import Sample

_REVISION = "9dace8acebf210929bd47256d7ded44439377946"
_URL = (
    "https://huggingface.co/datasets/visualwebbench/VisualWebBench/resolve/"
    f"{_REVISION}/element_ocr/test-00000-of-00001.parquet"
)
_SHA256 = "d57d0e2484add92bbfb1a9f56427b70c056986d9e33032c4156220eef76feeb4"
_OUT = Path(__file__).resolve().parent / "data" / "realweb"
_FRAME_REACH = 6


def fetch(target: Path, opener=urlopen) -> Path:
    """Download the pinned file to *target*; a file with another checksum is not kept."""
    with opener(_URL) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != _SHA256:
        raise ValueError(f"checksum mismatch for {_URL}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def _frame_depth(red_lines: np.ndarray) -> int:
    """Lines to cut from an edge: up to the last mostly red one near that edge."""
    hits = np.flatnonzero(red_lines[:_FRAME_REACH] > 0.3)
    return int(hits[-1]) + 1 if hits.size else 0


def strip_frame(image: Image.Image) -> Image.Image:
    """Cut the red frame the dataset draws along the edges of an element's box."""
    pixels = np.asarray(image.convert("RGB")).astype(int)
    red = (pixels[..., 0] > 180) & (pixels[..., 1] < 90) & (pixels[..., 2] < 90)
    rows, columns = red.mean(axis=1), red.mean(axis=0)
    top, bottom = _frame_depth(rows), _frame_depth(rows[::-1])
    left, right = _frame_depth(columns), _frame_depth(columns[::-1])
    return image.crop((left, top, image.width - right, image.height - bottom))


def crop(image: Image.Image, bbox: list[float]) -> Image.Image:
    """The element's box (fractions of the page: left, top, right, bottom) without its frame."""
    x1, y1, x2, y2 = bbox
    box = (round(x1 * image.width), round(y1 * image.height),
           round(x2 * image.width), round(y2 * image.height))  # fmt: skip
    return strip_frame(image.crop(box))


def generate(out_dir, parquet) -> list[Sample]:
    """Write one crop per row of *parquet* into *out_dir* and return their metadata."""
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for row in pq.read_table(parquet).to_pylist():
        path = out_dir / f"{row['id']}.png"
        with Image.open(io.BytesIO(row["image"]["bytes"])) as page:
            crop(page.convert("RGB"), row["bbox"]).save(path)
        samples.append(
            Sample(
                path=path,
                gt=row["answer"],
                source="visualwebbench",
                split="real_web",
                # Elements of one site share fonts and colours: the site is the cluster.
                text_id=row["website"],
                lang="en",
                content="web",
                font="na",
                font_size=0,
                theme="na",
                degradation="na",
            )
        )
    return samples


def main() -> None:
    parquet = _OUT / "element_ocr.parquet"
    if not parquet.exists():
        fetch(parquet)
    samples = generate(_OUT / "crops", parquet)
    manifest = [{**dataclasses.asdict(s), "path": str(s.path)} for s in samples]
    (_OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"Cropped {len(samples)} elements into {_OUT / 'crops'}")


if __name__ == "__main__":
    main()
