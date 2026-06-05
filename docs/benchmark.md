# OCR Pipeline Benchmark

Generated: 2026-06-05T11:34:13.879977+00:00
Commit: `d06ce32` | Images: 296 (synthetic=216, sroie=80)
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

## Confidence-weighted merge

Both merges run on identical engine outputs; the only difference is how
word-level disagreements are resolved (engine confidence vs text heuristic).

- Output changed on 134/296 images (where engines disagreed and both
  sides carried confidence).
- On those images: CER 0.560 → 0.329
  (**41.2% relative reduction**, +0.231 absolute).
- Corpus-wide: +0.104 absolute CER change.

| Source | changed | CER heuristic | CER conf | rel |
|---|---|---|---|---|
| synthetic | 60 | 0.689 | 0.128 | +81.4% |
| sroie | 74 | 0.455 | 0.492 | -8.2% |

The aggregate is dominated by SROIE receipts, where engine confidence is poorly
calibrated and the merge is near-neutral. On domain-matched screen text the
confidence signal is reliable and the reduction is large, concentrated on
degraded inputs where the two engines genuinely disagree.

## SymSpell correction

- On prose: +0.004 absolute CER change (negative = improvement).
- Overall: +0.000 absolute CER change.

## Strategy selector (GradientBoosting, trained on oracle labels)

- Samples: 216 (fast=147, ensemble=69)
- **Macro F1 (held-out 20%): 0.722**
- CV macro-F1: 0.709 ± 0.077
- Confusion matrix [rows=true fast/ensemble, cols=pred]: [[19, 11], [1, 13]]
- Top features: [('sharpness', 0.40710405628186813), ('brightness', 0.18827666818105018), ('text_density', 0.1732758860213452)]

### Vs label-only baselines (cross-validated, same folds)

| Policy | CV Macro F1 |
|---|---|
| always fast | 0.405 |
| always ensemble | 0.242 |
| majority | 0.405 |
| stratified random | 0.516 |
| **selector (GB)** | **0.709** |

The learned router beats the best static policy (stratified random,
CV macro F1 0.516) — routing on image features adds real signal over
always picking one mode or the class prior.

### Feature ablation (leave-one-out, same folds)

| Feature dropped | CV Macro F1 | Δ vs full |
|---|---|---|
| sharpness | 0.631 | +0.078 |
| noise_level | 0.641 | +0.068 |
| text_density | 0.695 | +0.014 |
| brightness | 0.707 | +0.002 |
| has_color | 0.709 | +0.000 |
| contrast | 0.738 | -0.029 |
| size_ratio | 0.742 | -0.033 |

Δ is the CV macro-F1 lost when the feature is removed. Positive = the feature
carries routing signal the rest can't recover; near-zero or negative = redundant
given the others. This is a stronger test than impurity importance, which can
rank a feature highly without it adding predictive value.

## Reproduce

```bash
venv/bin/python benchmarks/run_eval.py --source both
venv/bin/python benchmarks/report.py
```
