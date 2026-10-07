"""Run every candidate pipeline over the eval corpus.

Two passes over the same samples:
- accuracy (default): every pipeline on every image, in worker processes. Records
  the text, its CER and the confidence statistics of each pipeline.
- timing (--timing): one process, every pipeline on a development sample and on
  all evaluation slices. Nothing else should run on the machine meanwhile.

Sources: rendered screen text split by text into train / val / test plus an
unseen-font slice, photographed receipts (SROIE), and pages rendered by a browser.
Records are appended to a .partial.jsonl file, so an interrupted run continues
with --resume.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from PIL import Image

from benchmarks.corpus import load_items
from benchmarks.dataset import load_sroie
from benchmarks.metrics import cer, normalize_text
from benchmarks.pool import POOL
from benchmarks.synthetic import generate
from sniptext.analyzer import FEATURE_NAMES, ImageAnalyzer
from sniptext.pipelines import Pipeline, recognize
from sniptext.router import CONF_STAT_NAMES, conf_stats

LANGUAGE = "eng+rus"
DEV_SPLITS = ("train", "val")
EVAL_SLICES = ("test", "unseen_font", "ood", "browser")
_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results.json"
_PARTIAL = _HERE / "results.partial.jsonl"
_TIMING = _HERE / "timing.json"
_TIMING_PARTIAL = _HERE / "timing.partial.jsonl"
_SYNTH_DIR = _HERE / "data" / "synthetic"
_MANIFEST = _HERE / "data" / "browser" / "manifest.json"
_META = ("source", "split", "text_id", "lang", "content", "font", "font_size", "theme",
         "degradation", "scale", "gt")  # fmt: skip


def collect_samples(seed: int, sroie_limit: int, manifest=_MANIFEST) -> list[dict]:
    samples = [dataclasses.asdict(s) for s in generate(_SYNTH_DIR, load_items(seed), seed=seed)]
    for path, gt in load_sroie(limit=sroie_limit):
        samples.append(
            {
                "path": path,
                "gt": gt,
                "source": "sroie",
                "split": "ood",
                "text_id": path.name,
                "lang": "en",
                "content": "receipt",
                "font": "na",
                "font_size": 0,
                "theme": "na",
                "degradation": "na",
                "scale": 1.0,
            }  # fmt: skip
        )
    if Path(manifest).exists():
        samples += json.loads(Path(manifest).read_text())
    return samples


def run_pool(image: Image.Image, pool=POOL, lang: str = LANGUAGE) -> dict:
    """Text and word confidences of each pipeline; a pipeline that fails reads as empty."""
    outputs = {}
    for pipeline in pool:
        try:
            outputs[pipeline.name] = recognize(pipeline, image, lang)
        except Exception as e:
            print(f"[empty] {pipeline.name}: {e}", file=sys.stderr)
            outputs[pipeline.name] = ("", [])
    return outputs


def build_row(sample: dict, image: Image.Image, outputs: dict, analyzer: ImageAnalyzer) -> dict:
    """One image's record: metadata, routing features and each pipeline's result."""
    gt = normalize_text(sample["gt"])
    return {
        "image": Path(sample["path"]).name,
        **{key: sample[key] for key in _META},
        "pixels": image.width * image.height,
        "features": [float(v) for v in analyzer.extract_features(image)],
        "cer": {name: cer(normalize_text(text), gt) for name, (text, _) in outputs.items()},
        "text": {name: text for name, (text, _) in outputs.items()},
        "conf_stats": {
            name: [float(v) for v in conf_stats(confs)] for name, (_, confs) in outputs.items()
        },
    }


_ANALYZER = ImageAnalyzer()


def _quiet() -> None:
    # The app logs at INFO; per-call debug lines would be timed along with the OCR.
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def _init_worker() -> None:
    # One Tesseract thread per worker: the workers are the parallelism.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    _quiet()


def _accuracy_row(sample: dict) -> dict:
    with Image.open(sample["path"]) as opened:
        image = opened.convert("RGB")
    return build_row(sample, image, run_pool(image), _ANALYZER)


def _read_partial(path: Path, resume: bool) -> dict:
    done: dict[str, dict] = {}
    if resume and path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            done[record["image"]] = record
    else:
        path.write_text("")
    return done


def accuracy_pass(samples: list[dict], workers: int, resume: bool) -> None:
    done = _read_partial(_PARTIAL, resume)
    todo = [s for s in samples if Path(s["path"]).name not in done]
    with open(_PARTIAL, "a") as partial, Pool(workers, initializer=_init_worker) as pool:
        for count, row in enumerate(pool.imap_unordered(_accuracy_row, todo, chunksize=4), 1):
            partial.write(json.dumps(row) + "\n")
            partial.flush()
            done[row["image"]] = row
            if count % 200 == 0:
                print(f"processed {count}/{len(todo)}", file=sys.stderr)
    order = [Path(s["path"]).name for s in samples]
    _RESULTS.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "conf_stat_names": list(CONF_STAT_NAMES),
                "language": LANGUAGE,
                "pipelines": [p.to_dict() for p in POOL],
                "rows": [done[name] for name in order],
            }
        )
    )
    print(f"Wrote {len(order)} rows to {_RESULTS}")


def timing_images(samples: list[dict], seed: int, n_dev: int = 300) -> list[dict]:
    """A fixed random sample of development images plus every evaluation image."""
    dev = [s for s in samples if s["split"] in DEV_SPLITS]
    chosen = random.Random(seed).sample(dev, min(n_dev, len(dev)))
    return chosen + [s for s in samples if s["split"] in EVAL_SLICES]


def _timed(pipeline: Pipeline, image: Image.Image) -> float:
    start = time.perf_counter()
    try:
        recognize(pipeline, image, LANGUAGE)
    except Exception as e:
        print(f"[failed] {pipeline.name}: {e}", file=sys.stderr)
    return time.perf_counter() - start


def timing_pass(samples: list[dict], seed: int, resume: bool) -> None:
    chosen = timing_images(samples, seed)
    done = _read_partial(_TIMING_PARTIAL, resume)
    with Image.open(chosen[0]["path"]) as opened:
        run_pool(opened.convert("RGB"))  # the first call of a process is slower
    loadavg_start = list(os.getloadavg())
    with open(_TIMING_PARTIAL, "a") as partial:
        for count, sample in enumerate(chosen, 1):
            name = Path(sample["path"]).name
            if name in done:
                continue
            with Image.open(sample["path"]) as opened:
                image = opened.convert("RGB")
            start = time.perf_counter()
            _ANALYZER.extract_features(image)
            record = {
                "image": name,
                "features": time.perf_counter() - start,
                "time": {pipeline.name: _timed(pipeline, image) for pipeline in POOL},
            }
            partial.write(json.dumps(record) + "\n")
            partial.flush()
            done[name] = record
            if count % 100 == 0:
                print(f"timed {count}/{len(chosen)}", file=sys.stderr)
    _TIMING.write_text(
        json.dumps(
            {
                "loadavg_start": loadavg_start,
                "loadavg_end": list(os.getloadavg()),
                "rows": {Path(s["path"]).name: done[Path(s["path"]).name] for s in chosen},
            }
        )
    )
    print(f"Wrote {len(chosen)} timings to {_TIMING}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sroie-limit", type=int, default=80)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timing", action="store_true", help="run the timing pass instead")
    parser.add_argument("--max-images", type=int, default=None, help="debug: cap the run")
    parser.add_argument("--resume", action="store_true", help="continue the partial file")
    args = parser.parse_args()

    _quiet()
    samples = collect_samples(args.seed, args.sroie_limit)
    if args.max_images:
        samples = samples[: args.max_images]
    if args.timing:
        timing_pass(samples, args.seed, args.resume)
    else:
        accuracy_pass(samples, args.workers, args.resume)


if __name__ == "__main__":
    main()
