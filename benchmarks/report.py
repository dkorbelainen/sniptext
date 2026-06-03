"""Aggregate results.json + selector metrics into docs/benchmark.md."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.train_selector import train_and_report

_RESULTS = Path(__file__).resolve().parent / "results.json"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPORT = _REPO_ROOT / "docs" / "benchmark.md"


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _mean(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def _variant_table(rows) -> str:
    if not rows:
        return "_(no samples)_\n"
    return (
        "| Variant | CER | WER |\n|---|---|---|\n"
        f"| Tesseract only | {_mean(rows, 'cer_tesseract'):.3f} | {_mean(rows, 'wer_tesseract'):.3f} |\n"
        f"| EasyOCR only | {_mean(rows, 'cer_easyocr'):.3f} | {_mean(rows, 'wer_easyocr'):.3f} |\n"
        f"| Ensemble merge | {_mean(rows, 'cer_ensemble'):.3f} | {_mean(rows, 'wer_ensemble'):.3f} |\n"
        f"| Ensemble + SymSpell | {_mean(rows, 'cer_ensemble_corrected'):.3f} | — |\n"
    )


def main():
    rows = json.loads(_RESULTS.read_text())
    n = len(rows)

    def subset(**kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    synth = subset(source="synthetic")
    sroie = subset(source="sroie")

    # Conditional ensemble gain: only on images where the ensemble actually wins.
    ens_rows = [r for r in rows if r["oracle_label"] == "ensemble"]
    if ens_rows:
        cer_fast_on_ens = _mean(ens_rows, "cer_tesseract")
        cer_ens_on_ens = _mean(ens_rows, "cer_ensemble")
        cond_gain_rel = (
            (cer_fast_on_ens - cer_ens_on_ens) / cer_fast_on_ens * 100
            if cer_fast_on_ens
            else 0.0
        )
    else:
        cer_fast_on_ens = cer_ens_on_ens = cond_gain_rel = float("nan")

    # SymSpell gain on prose (real-word text where a dictionary helps).
    prose = subset(content="prose")
    if prose:
        symspell_prose = _mean(prose, "cer_ensemble") - _mean(prose, "cer_ensemble_corrected")
    else:
        symspell_prose = float("nan")
    symspell_overall = _mean(rows, "cer_ensemble") - _mean(rows, "cer_ensemble_corrected")

    sel = train_and_report()
    macro_f1 = sel["classification_report"]["macro avg"]["f1-score"]
    top_feats = sorted(sel["feature_importance"].items(), key=lambda x: -x[1])[:3]

    # Per-difficulty (synthetic) table.
    diff_lines = ["| Difficulty | n | CER Tesseract | CER Ensemble |", "|---|---|---|---|"]
    for diff in ("clean", "medium", "heavy"):
        d = subset(source="synthetic", difficulty=diff)
        if d:
            diff_lines.append(
                f"| {diff} | {len(d)} | {_mean(d, 'cer_tesseract'):.3f} | {_mean(d, 'cer_ensemble'):.3f} |"
            )

    # Theme (synthetic) table.
    theme_lines = ["| Theme | n | CER Tesseract | CER Ensemble |", "|---|---|---|---|"]
    for theme in ("light", "dark"):
        t = subset(source="synthetic", theme=theme)
        if t:
            theme_lines.append(
                f"| {theme} | {len(t)} | {_mean(t, 'cer_tesseract'):.3f} | {_mean(t, 'cer_ensemble'):.3f} |"
            )

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# OCR Pipeline Benchmark

Generated: {datetime.now(timezone.utc).isoformat()}
Commit: `{_git_commit()}` | Images: {n} (synthetic={len(synth)}, sroie={len(sroie)})
Metric: mean per-sample CER/WER, whitespace-normalized (case preserved). Lower is better.

## Overall accuracy

{_variant_table(rows)}

## Synthetic (domain-matched screen text)

{_variant_table(synth)}

### By difficulty

{chr(10).join(diff_lines)}

### By theme

{chr(10).join(theme_lines)}

## SROIE (hard real-world receipts)

{_variant_table(sroie)}

## Ensemble — conditional gain

On the {len(ens_rows)} images where the ensemble beats the fast path, it
reduces CER from {cer_fast_on_ens:.3f} to {cer_ens_on_ens:.3f}
(**{cond_gain_rel:.1f}% relative reduction**). On easy/clean inputs the
selector routes to the fast path, so the ensemble cost is avoided.

## SymSpell correction

- On prose: {symspell_prose:+.3f} absolute CER change (negative = improvement).
- Overall: {symspell_overall:+.3f} absolute CER change.

## Strategy selector (GradientBoosting, trained on oracle labels)

- Samples: {sel['n_samples']} (fast={sel['label_distribution']['fast']}, ensemble={sel['label_distribution']['ensemble']})
- **Macro F1 (held-out 20%): {macro_f1:.3f}**
- CV macro-F1: {sel['cv_f1_macro_mean']:.3f} ± {sel['cv_f1_macro_std']:.3f}
- Confusion matrix [rows=true fast/ensemble, cols=pred]: {sel['confusion_matrix']}
- Top features: {top_feats}

## Reproduce

```bash
venv/bin/python benchmarks/run_eval.py --source both
venv/bin/python benchmarks/report.py
```
"""
    _REPORT.write_text(md)
    print(f"Wrote report to {_REPORT}")


if __name__ == "__main__":
    main()
