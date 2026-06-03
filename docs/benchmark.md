# OCR Pipeline Benchmark

Generated: 2026-06-03T21:12:25.844422+00:00
Commit: `fc80010` | Images: 296 (synthetic=216, sroie=80)
Metric: mean per-sample CER/WER, whitespace-normalized (case preserved). Lower is better.

## Overall accuracy

| Variant | CER | WER |
|---|---|---|
| Tesseract only | 0.451 | 0.767 |
| EasyOCR only | 0.282 | 0.480 |
| Ensemble merge | 0.344 | 0.636 |
| Ensemble + SymSpell | 0.344 | — |


## Synthetic (domain-matched screen text)

| Variant | CER | WER |
|---|---|---|
| Tesseract only | 0.482 | 0.809 |
| EasyOCR only | 0.161 | 0.372 |
| Ensemble merge | 0.300 | 0.615 |
| Ensemble + SymSpell | 0.298 | — |


### By difficulty

| Difficulty | n | CER Tesseract | CER Ensemble |
|---|---|---|---|
| clean | 72 | 0.002 | 0.045 |
| medium | 72 | 0.195 | 0.117 |
| heavy | 72 | 1.248 | 0.738 |

### By theme

| Theme | n | CER Tesseract | CER Ensemble |
|---|---|---|---|
| light | 108 | 0.263 | 0.114 |
| dark | 108 | 0.701 | 0.486 |

## SROIE (hard real-world receipts)

| Variant | CER | WER |
|---|---|---|
| Tesseract only | 0.369 | 0.654 |
| EasyOCR only | 0.609 | 0.774 |
| Ensemble merge | 0.464 | 0.693 |
| Ensemble + SymSpell | 0.467 | — |


## Ensemble — conditional gain

On the 75 images where the ensemble beats the fast path, it
reduces CER from 0.849 to 0.245
(**71.2% relative reduction**). On easy/clean inputs the
selector routes to the fast path, so the ensemble cost is avoided.

## SymSpell correction

- On prose: +0.004 absolute CER change (negative = improvement).
- Overall: +0.000 absolute CER change.

## Strategy selector (GradientBoosting, trained on oracle labels)

- Samples: 216 (fast=147, ensemble=69)
- **Macro F1 (held-out 20%): 0.722**
- CV macro-F1: 0.701 ± 0.030
- Confusion matrix [rows=true fast/ensemble, cols=pred]: [[19, 11], [1, 13]]
- Top features: [('sharpness', 0.40710405628186813), ('brightness', 0.18827666818105018), ('text_density', 0.1732758860213452)]

## Reproduce

```bash
venv/bin/python benchmarks/run_eval.py --source both
venv/bin/python benchmarks/report.py
```
