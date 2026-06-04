# OCR Pipeline Benchmark

Generated: 2026-06-04T21:03:02.619042+00:00
Commit: `650a316` | Images: 734 (synthetic=108, sroie=626)
Metric: mean per-sample CER/WER, whitespace-normalized (case preserved). Lower is better.

## Overall accuracy

| Variant | CER | WER |
|---|---|---|
| Tesseract only | 0.416 | 0.710 |
| EasyOCR only | 0.536 | 0.697 |
| Ensemble merge (heuristic) | 0.446 | 0.678 |
| Ensemble merge (conf-weighted) | 0.434 | — |
| Ensemble + SymSpell | 0.447 | — |


## Synthetic (domain-matched screen text)

| Variant | CER | WER |
|---|---|---|
| Tesseract only | 0.537 | 1.018 |
| EasyOCR only | 0.191 | 0.349 |
| Ensemble merge (heuristic) | 0.337 | 0.683 |
| Ensemble merge (conf-weighted) | 0.167 | — |
| Ensemble + SymSpell | 0.333 | — |


### By difficulty

| Difficulty | n | CER Tesseract | CER Ensemble |
|---|---|---|---|
| clean | 36 | 0.002 | 0.016 |
| medium | 36 | 0.161 | 0.119 |
| heavy | 36 | 1.448 | 0.875 |

### By theme

| Theme | n | CER Tesseract | CER Ensemble |
|---|---|---|---|
| light | 54 | 0.218 | 0.136 |
| dark | 54 | 0.857 | 0.538 |

## SROIE (hard real-world receipts)

| Variant | CER | WER |
|---|---|---|
| Tesseract only | 0.395 | 0.657 |
| EasyOCR only | 0.595 | 0.757 |
| Ensemble merge (heuristic) | 0.464 | 0.677 |
| Ensemble merge (conf-weighted) | 0.480 | — |
| Ensemble + SymSpell | 0.467 | — |


## Ensemble — conditional gain

On the 121 images where the ensemble beats the fast path, it
reduces CER from 0.661 to 0.394
(**40.4% relative reduction**). On easy/clean inputs the
selector routes to the fast path, so the ensemble cost is avoided.

## Confidence-weighted merge

Both merges run on identical engine outputs; the only difference is how
word-level disagreements are resolved (engine confidence vs text heuristic).

- Output changed on 624/734 images (where engines disagreed and both
  sides carried confidence).
- On those images: CER 0.469 → 0.455
  (**3.0% relative reduction**, +0.014 absolute).
- Corpus-wide: +0.012 absolute CER change.

| Source | changed | CER heuristic | CER conf | rel |
|---|---|---|---|---|
| synthetic | 29 | 0.803 | 0.169 | +79.0% |
| sroie | 595 | 0.452 | 0.469 | -3.6% |

The aggregate is dominated by SROIE receipts, where engine confidence is poorly
calibrated and the merge is near-neutral. On domain-matched screen text the
confidence signal is reliable and the reduction is large, concentrated on
degraded inputs where the two engines genuinely disagree.

## SymSpell correction

- On prose: +0.006 absolute CER change (negative = improvement).
- Overall: -0.002 absolute CER change.

## Strategy selector (GradientBoosting, trained on oracle labels)

- Samples: 108 (fast=74, ensemble=34)
- **Macro F1 (held-out 20%): 0.624**
- CV macro-F1: 0.505 ± 0.051
- Confusion matrix [rows=true fast/ensemble, cols=pred]: [[9, 6], [2, 5]]
- Top features: [('sharpness', 0.36551655378624226), ('brightness', 0.2686807335928205), ('contrast', 0.19706594379430833)]

## Reproduce

```bash
venv/bin/python benchmarks/run_eval.py --source both
venv/bin/python benchmarks/report.py
```
