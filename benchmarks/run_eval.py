"""Run every router action over the eval corpus and record per-image results.

Sources:
- synthetic: rendered screen text, split by text into train / val / test, plus an
  unseen-font slice
- sroie: photographed receipts, kept as an out-of-domain slice

Records are appended to results.partial.jsonl as they are produced, so an
interrupted run continues with --resume.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from benchmarks.corpus import load_items
from benchmarks.dataset import load_sroie
from benchmarks.engines import EngineRunner, RunResult, _flat_word_conf
from benchmarks.legacy_policy import legacy_features
from benchmarks.metrics import cer, normalize_text, wer
from benchmarks.synthetic import generate
from sniptext.analyzer import FEATURE_NAMES, ImageAnalyzer
from sniptext.config import Config
from sniptext.metrics import OCRQualityMetrics
from sniptext.ocr import EasyOCRBackend
from sniptext.router import CONF_STAT_NAMES, conf_stats

LANGUAGE = "eng+rus"
_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results.json"
_WORD_CONF = _HERE / "word_conf.json"
_MERGE_INPUTS = _HERE / "merge_inputs.json"
_PARTIAL = _HERE / "results.partial.jsonl"
_TIMING_CPU = _HERE / "timing_cpu.json"
_SYNTH_DIR = _HERE / "data" / "synthetic"
_ACTION_TEXTS = ("tesseract_plain", "tesseract", "easyocr", "merge")


def _label_correct(rec_words, gt_words):
    """Per-recognized-word correctness via sequence alignment to GT tokens.

    A recognized word counts as correct only if it survives in an 'equal'
    block of the optimal alignment; substitutions/deletions are incorrect.
    This yields the (confidence, correct) pairs a reliability diagram needs.
    """
    labels = [0] * len(rec_words)
    sm = difflib.SequenceMatcher(None, rec_words, gt_words, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                labels[i] = 1
    return labels


def build_row(sample: dict, image: Image.Image, result: RunResult, analyzer, quality):
    """One image's record, its merge-replay input and its (confidence, correct) word pairs."""
    name = Path(sample["path"]).name
    gt = normalize_text(sample["gt"])
    normalized = {key: normalize_text(result.texts[key]) for key in _ACTION_TEXTS}

    start = time.perf_counter()
    features = analyzer.extract_features(image)
    t_features = time.perf_counter() - start

    # The label the app used to record for online retraining: quality score of
    # the fast text (with Tesseract confidences on a 0-100 scale) against the
    # quality score of the merged text (without confidences).
    tess_conf_percent = [c * 100.0 for line in (result.confs["tesseract"] or []) for c in line]
    weak = {
        "fast_quality": quality.calculate_quality_score(
            result.texts["tesseract_plain"], tess_conf_percent or None
        ),
        "ens_quality": quality.calculate_quality_score(result.texts["merge"], None),
    }

    row = {
        "image": name,
        **{key: sample[key] for key in ("source", "split", "text_id", "lang", "content")},
        **{key: sample[key] for key in ("font", "font_size", "theme", "degradation")},
        "features": [float(v) for v in features],
        "legacy_features": [float(v) for v in legacy_features(image)],
        "conf_stats": [float(v) for v in conf_stats(result.confs["tesseract"])],
        "cer": {key: cer(normalized[key], gt) for key in _ACTION_TEXTS},
        "wer": {key: wer(normalized[key], gt) for key in _ACTION_TEXTS},
        "time": {**result.times, "features": t_features},
        "weak": weak,
    }
    merge_input = {
        "image": name,
        "source": sample["source"],
        "split": sample["split"],
        "text_id": sample["text_id"],
        "gt": sample["gt"],
        "tess_text": result.texts["tesseract"],
        "tess_conf": result.confs["tesseract"],
        "easy_text": result.texts["easyocr"],
        "easy_conf": result.confs["easyocr"],
    }
    word_conf = []
    gt_words = gt.split()
    for engine in ("tesseract", "easyocr"):
        pairs = _flat_word_conf(result.texts[engine], result.confs[engine])
        labels = _label_correct([w for w, _ in pairs], gt_words)
        for (_, conf), correct in zip(pairs, labels):
            word_conf.append(
                {
                    "conf": conf,
                    "correct": correct,
                    "source": sample["source"],
                    "engine": engine,
                    "image": name,
                }
            )
    return row, merge_input, word_conf


def _collect_samples(seed: int, sroie_limit: int) -> list[dict]:
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
            }
        )
    return samples


def _time_easyocr_on_cpu(samples: list[dict], gpu_times: dict, count: int, seed: int) -> dict:
    """Mean EasyOCR time on CPU and on GPU for the same random synthetic images."""
    pool = [s for s in samples if s["source"] == "synthetic"]
    chosen = random.Random(seed).sample(pool, min(count, len(pool)))
    backend = EasyOCRBackend(Config(ocr_language=LANGUAGE, use_gpu=False))
    backend.recognize_detailed(Image.open(chosen[0]["path"]).convert("RGB"))
    cpu = []
    for sample in chosen:
        image = Image.open(sample["path"]).convert("RGB")
        start = time.perf_counter()
        backend.recognize_detailed(image)
        cpu.append(time.perf_counter() - start)
    gpu = [gpu_times[Path(s["path"]).name] for s in chosen]
    return {
        "n": len(chosen),
        "easyocr_cpu_mean": sum(cpu) / len(cpu),
        "easyocr_gpu_mean": sum(gpu) / len(gpu),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sroie-limit", type=int, default=80)
    parser.add_argument("--cpu-timing", type=int, default=200, help="images timed on CPU; 0 skips")
    parser.add_argument("--max-images", type=int, default=None, help="debug: cap the run")
    parser.add_argument("--resume", action="store_true", help="continue results.partial.jsonl")
    args = parser.parse_args()

    samples = _collect_samples(args.seed, args.sroie_limit)
    if args.max_images:
        samples = samples[: args.max_images]

    done: dict[str, dict] = {}
    if args.resume and _PARTIAL.exists():
        for line in _PARTIAL.read_text().splitlines():
            record = json.loads(line)
            done[record["row"]["image"]] = record
    else:
        _PARTIAL.write_text("")

    runner = EngineRunner(Config(ocr_language=LANGUAGE))
    analyzer = ImageAnalyzer()
    quality = OCRQualityMetrics()
    # Load models and warm caches so the first timed image is not an outlier.
    runner.run_all(Image.open(samples[0]["path"]).convert("RGB"))

    with open(_PARTIAL, "a") as partial:
        for position, sample in enumerate(samples, 1):
            name = Path(sample["path"]).name
            if name in done:
                continue
            image = Image.open(sample["path"]).convert("RGB")
            try:
                result = runner.run_all(image)
            except Exception as e:
                print(f"[skip] {name}: {e}", file=sys.stderr)
                continue
            row, merge_input, word_conf = build_row(sample, image, result, analyzer, quality)
            record = {"row": row, "merge_input": merge_input, "word_conf": word_conf}
            partial.write(json.dumps(record) + "\n")
            partial.flush()
            done[name] = record
            if position % 100 == 0:
                print(f"processed {position}/{len(samples)}", file=sys.stderr)

    order = [Path(s["path"]).name for s in samples]
    records = [done[name] for name in order if name in done]
    _RESULTS.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "conf_stat_names": list(CONF_STAT_NAMES),
                "language": LANGUAGE,
                "rows": [r["row"] for r in records],
            }
        )
    )
    _MERGE_INPUTS.write_text(json.dumps([r["merge_input"] for r in records]))
    _WORD_CONF.write_text(json.dumps([w for r in records for w in r["word_conf"]]))
    print(f"Wrote {len(records)} rows to {_RESULTS} ({len(samples) - len(records)} skipped)")

    if args.cpu_timing:
        gpu_times = {r["row"]["image"]: r["row"]["time"]["easyocr"] for r in records}
        timed = [s for s in samples if Path(s["path"]).name in gpu_times]
        timing = _time_easyocr_on_cpu(timed, gpu_times, args.cpu_timing, args.seed)
        _TIMING_CPU.write_text(json.dumps(timing, indent=2))
        print(f"Wrote {_TIMING_CPU}: {timing}")


if __name__ == "__main__":
    main()
