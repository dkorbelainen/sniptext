"""Train a GradientBoosting strategy selector on oracle labels from results.json."""

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils.class_weight import compute_class_weight

_RESULTS = Path(__file__).resolve().parent / "results.json"
_FEATURE_NAMES = [
    "brightness",
    "contrast",
    "sharpness",
    "has_color",
    "size_ratio",
    "text_density",
    "noise_level",
]


def _baselines(y: np.ndarray, splitter) -> Dict[str, float]:
    """Cross-validated macro-F1 of label-only policies, on the same folds as
    the selector's CV score.

    A single 20% split is too small here to compare against (val ~20 samples),
    so baselines are averaged over the CV folds to match cv_f1_macro_mean. A
    learned router only earns its keep if it beats the best static policy
    (always-fast / always-ensemble / majority) and chance.
    """
    if splitter is None:
        return {}
    rng = np.random.default_rng(42)
    acc: Dict[str, list] = {
        k: [] for k in ("always_fast", "always_ensemble", "majority", "stratified_random")
    }

    def mf1(yv: np.ndarray, pred: np.ndarray) -> float:
        return float(f1_score(yv, pred, average="macro", zero_division=0))

    for tr_idx, val_idx in splitter.split(y, y):
        y_tr, y_val = y[tr_idx], y[val_idx]
        n = len(y_val)
        majority = int(np.bincount(y_tr).argmax())
        p_ens = float((y_tr == 1).mean())
        random_pred = (rng.random(n) < p_ens).astype(int)
        acc["always_fast"].append(mf1(y_val, np.zeros(n, dtype=int)))
        acc["always_ensemble"].append(mf1(y_val, np.ones(n, dtype=int)))
        acc["majority"].append(mf1(y_val, np.full(n, majority, dtype=int)))
        acc["stratified_random"].append(mf1(y_val, random_pred))

    return {k: float(np.mean(v)) for k, v in acc.items()}


def train_and_report() -> Dict:
    all_rows = json.loads(_RESULTS.read_text())
    # The selector runs on SnipText's real traffic: screen captures, modelled
    # by the synthetic corpus. SROIE receipts are an out-of-domain OCR-accuracy
    # slice that routes uniformly to the fast path and adds no routing signal,
    # so they are excluded from selector training.
    rows = [r for r in all_rows if r.get("source") == "synthetic"] or all_rows
    X = np.array([r["features"] for r in rows])
    y = np.array([0 if r["oracle_label"] == "fast" else 1 for r in rows])

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Only one class present in oracle labels ({int(y.sum())} ensemble / "
            f"{int(len(y) - y.sum())} fast); cannot train a selector. "
            "Add more degraded/synthetic samples or adjust --margin."
        )

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Balance the fast/ensemble classes: ensemble wins are the minority, and an
    # unweighted model collapses to always predicting fast.
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    weight_map = dict(zip(classes, weights))
    sample_weight = np.array([weight_map[c] for c in y_tr])

    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    y_pred = model.predict(X_val)

    report = classification_report(
        y_val,
        y_pred,
        target_names=["fast", "ensemble"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_val, y_pred).tolist()

    n_folds = min(5, int(np.bincount(y).min()))
    # Shuffled folds: the synthetic corpus is generated combo-by-combo, so
    # unshuffled folds would group same-theme/difficulty samples and bias CV.
    # The same splitter feeds the baselines for an apples-to-apples comparison.
    splitter = (
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42) if n_folds >= 2 else None
    )
    cv = cross_val_score(model, X, y, cv=splitter, scoring="f1_macro") if splitter else None

    importances = dict(zip(_FEATURE_NAMES, model.feature_importances_.tolist()))

    out = {
        "baselines": _baselines(y, splitter),
        "n_samples": len(rows),
        "label_distribution": {"fast": int((y == 0).sum()), "ensemble": int((y == 1).sum())},
        "classification_report": report,
        "confusion_matrix": cm,
        "cv_f1_macro_mean": float(cv.mean()) if cv is not None else None,
        "cv_f1_macro_std": float(cv.std()) if cv is not None else None,
        "feature_importance": importances,
    }
    return out


if __name__ == "__main__":
    print(json.dumps(train_and_report(), indent=2))
