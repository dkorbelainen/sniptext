"""Fit the router's regressors with scikit-learn and export them for numpy inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sniptext.router import FORMAT


def build_regressor(spec: dict):
    """One unfitted regressor for a model spec."""
    kind = spec.get("kind")
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"]))
    if kind == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=spec["n_estimators"],
            max_depth=spec["max_depth"],
            learning_rate=spec["learning_rate"],
            random_state=0,
        )
    raise ValueError(f"unknown model kind {kind!r}")


def fit(spec: dict, X, y) -> list:
    """One fitted regressor per column of *y*."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return [build_regressor(spec).fit(X, y[:, k]) for k in range(y.shape[1])]


def predict(models: list, X) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.clip(np.column_stack([m.predict(X) for m in models]), 0.0, 1.0)


def _export_trees(model) -> dict:
    trees = [stage[0].tree_ for stage in model.estimators_]
    width = max(tree.node_count for tree in trees)

    def padded(values, fill):
        return [list(row) + [fill] * (width - len(row)) for row in values]

    return {
        "kind": "gbr",
        "init": float(model.init_.constant_[0][0]),
        "rate": float(model.learning_rate),
        "depth": int(max(tree.max_depth for tree in trees)),
        # scikit-learn marks a leaf with feature -2 and child -1.
        "feature": padded([[int(f) if f >= 0 else -1 for f in t.feature] for t in trees], -1),
        "threshold": padded([[float(v) for v in t.threshold] for t in trees], 0.0),
        "left": padded([[max(int(c), 0) for c in t.children_left] for t in trees], 0),
        "right": padded([[max(int(c), 0) for c in t.children_right] for t in trees], 0),
        "value": padded([[float(v) for v in t.value[:, 0, 0]] for t in trees], 0.0),
    }


def _export_ridge(model) -> dict:
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
    return {
        "kind": "ridge",
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in ridge.coef_],
        "intercept": float(ridge.intercept_),
    }


def export_model(policy, feature_names, pipelines, costs, time_weight, spec, models) -> dict:
    """The model file's content. *pipelines* are the actions, default first."""
    exporters = {"gbr": _export_trees, "ridge": _export_ridge}
    return {
        "format": FORMAT,
        "policy": policy,
        "feature_names": list(feature_names),
        "actions": [p.to_dict() for p in pipelines],
        "costs": [float(c) for c in costs],
        "time_weight": float(time_weight),
        "spec": spec,
        "regressors": [exporters[spec["kind"]](m) for m in models],
    }


def write_model(path, model: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, separators=(",", ":"), sort_keys=True) + "\n")
