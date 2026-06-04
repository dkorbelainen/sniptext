"""Run every OCR variant over the eval datasets, record per-image metrics.

Sources:
- synthetic : domain-matched clean/degraded screen text (primary)
- sroie     : real photographed receipts (hard real-world slice)
- both      : synthetic + sroie

Each row carries metadata (source/theme/difficulty/content) and an oracle
label: "ensemble" if the ensemble beats the fast (Tesseract-only) path on
that image, else "fast".
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from benchmarks.dataset import load_sroie
from benchmarks.engines import EngineRunner
from benchmarks.metrics import cer, normalize_text, wer
from benchmarks.synthetic import generate
from sniptext.analyzer import ImageAnalyzer
from sniptext.config import Config

_MARGIN = 0.0
_RESULTS = Path(__file__).resolve().parent / "results.json"
_SYNTH_DIR = Path(__file__).resolve().parent / "data" / "synthetic"


def _collect_samples(source: str, limit, n_per_combo, seed):
    """Yield unified sample dicts {path, gt, source, theme, difficulty, content}."""
    if source in ("synthetic", "both"):
        for s in generate(_SYNTH_DIR, n_per_combo=n_per_combo, seed=seed):
            yield {
                "path": s.path,
                "gt": s.gt,
                "source": "synthetic",
                "theme": s.theme,
                "difficulty": s.difficulty,
                "content": s.content,
            }
    if source in ("sroie", "both"):
        for img_path, gt in load_sroie(limit=limit):
            yield {
                "path": img_path,
                "gt": gt,
                "source": "sroie",
                "theme": "na",
                "difficulty": "hard",
                "content": "receipt",
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "sroie", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="cap on SROIE images")
    parser.add_argument("--n-per-combo", type=int, default=6, help="synthetic samples per combo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=_MARGIN)
    args = parser.parse_args()

    cfg = Config(ocr_language="eng")
    runner = EngineRunner(cfg)
    analyzer = ImageAnalyzer()

    rows = []
    samples = list(_collect_samples(args.source, args.limit, args.n_per_combo, args.seed))
    for i, sample in enumerate(samples):
        img_path, gt = sample["path"], sample["gt"]
        try:
            texts = runner.run_all(img_path)
        except Exception as e:
            print(f"[skip] {img_path.name}: {e}", file=sys.stderr)
            continue

        gt_n = normalize_text(gt)
        features = analyzer.extract_features(Image.open(img_path).convert("RGB"))
        cer_tess = cer(normalize_text(texts["tesseract"]), gt_n)
        cer_easy = cer(normalize_text(texts["easyocr"]), gt_n)
        cer_ens = cer(normalize_text(texts["ensemble"]), gt_n)
        cer_ens_conf = cer(normalize_text(texts["ensemble_conf"]), gt_n)
        cer_ens_corr = cer(normalize_text(texts["ensemble_corrected"]), gt_n)

        oracle = "ensemble" if (cer_tess - cer_ens) > args.margin else "fast"

        rows.append(
            {
                "image": img_path.name,
                "source": sample["source"],
                "theme": sample["theme"],
                "difficulty": sample["difficulty"],
                "content": sample["content"],
                "features": features.tolist(),
                "cer_tesseract": cer_tess,
                "cer_easyocr": cer_easy,
                "cer_ensemble": cer_ens,
                "cer_ensemble_conf": cer_ens_conf,
                "cer_ensemble_corrected": cer_ens_corr,
                "conf_changed_output": normalize_text(texts["ensemble"])
                != normalize_text(texts["ensemble_conf"]),
                "wer_tesseract": wer(normalize_text(texts["tesseract"]), gt_n),
                "wer_easyocr": wer(normalize_text(texts["easyocr"]), gt_n),
                "wer_ensemble": wer(normalize_text(texts["ensemble"]), gt_n),
                "oracle_label": oracle,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"processed {i + 1}/{len(samples)}...", file=sys.stderr)

    _RESULTS.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows to {_RESULTS}")


if __name__ == "__main__":
    main()
