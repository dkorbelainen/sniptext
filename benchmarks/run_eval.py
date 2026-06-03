"""Run every OCR variant over SROIE, record per-image metrics + oracle label."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from sniptext.analyzer import ImageAnalyzer
from sniptext.config import Config

from benchmarks.dataset import load_sroie
from benchmarks.engines import EngineRunner
from benchmarks.metrics import cer, wer

_MARGIN = 0.0  # ensemble must strictly beat the fast path to be worth it
_RESULTS = Path(__file__).resolve().parent / "results.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--margin", type=float, default=_MARGIN)
    args = parser.parse_args()

    cfg = Config(ocr_language="eng")
    runner = EngineRunner(cfg)
    analyzer = ImageAnalyzer()

    rows = []
    for i, (img_path, gt) in enumerate(load_sroie(limit=args.limit)):
        try:
            texts = runner.run_all(img_path)
        except Exception as e:  # keep the run going; record the failure
            print(f"[skip] {img_path.name}: {e}", file=sys.stderr)
            continue

        features = analyzer.extract_features(Image.open(img_path).convert("RGB"))
        cer_tess = cer(texts["tesseract"], gt)
        cer_easy = cer(texts["easyocr"], gt)
        cer_ens = cer(texts["ensemble"], gt)
        cer_ens_corr = cer(texts["ensemble_corrected"], gt)

        oracle = "ensemble" if (cer_tess - cer_ens) > args.margin else "fast"

        rows.append(
            {
                "image": img_path.name,
                "features": features.tolist(),
                "cer_tesseract": cer_tess,
                "cer_easyocr": cer_easy,
                "cer_ensemble": cer_ens,
                "cer_ensemble_corrected": cer_ens_corr,
                "wer_tesseract": wer(texts["tesseract"], gt),
                "wer_easyocr": wer(texts["easyocr"], gt),
                "wer_ensemble": wer(texts["ensemble"], gt),
                "oracle_label": oracle,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"processed {i + 1} images...", file=sys.stderr)

    _RESULTS.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows to {_RESULTS}")


if __name__ == "__main__":
    main()
