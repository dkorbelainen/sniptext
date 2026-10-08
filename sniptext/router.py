"""Per-image choice of a Tesseract pipeline.

The policy: predict the character error rate of each pipeline for this image,
then take the one minimising predicted CER + time_weight * seconds. The
regressors come from a model file shipped with the package and are evaluated
with numpy, so the model the app runs is the one the benchmark scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from .pipelines import DEFAULT, Pipeline

FORMAT = 1
POLICIES = ("static", "pre_ocr", "cascade")
CONF_STAT_NAMES = (
    "tess_conf_mean",
    "tess_conf_min",
    "tess_conf_p10",
    "tess_low_share",
    "tess_n_words",
)
_LOW_CONFIDENCE = 0.6
_MODEL_PATH = Path(__file__).resolve().parent / "data" / "router_model.json"


def conf_stats(confs: list[list[float]] | None) -> np.ndarray:
    """Summary of Tesseract's per-word confidences, in CONF_STAT_NAMES order."""
    flat = [c for line in (confs or []) for c in line]
    if not flat:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    values = np.asarray(flat, dtype=float)
    return np.array(
        [
            values.mean(),
            values.min(),
            np.percentile(values, 10),
            (values < _LOW_CONFIDENCE).mean(),
            min(len(values) / 100.0, 1.0),
        ]
    )


def choose_actions(pred_cer: np.ndarray, costs: np.ndarray, time_weight: float) -> np.ndarray:
    """Per row, the index of the action minimising predicted CER + time_weight * cost.

    Ties go to the cheaper action, then to the earlier one.
    """
    costs = np.asarray(costs, dtype=float)
    objective = np.asarray(pred_cer, dtype=float) + time_weight * costs
    tied = objective <= objective.min(axis=1, keepdims=True) + 1e-9
    return np.where(tied, np.broadcast_to(costs, objective.shape), np.inf).argmin(axis=1)


class _Trees:
    """Gradient-boosted regression trees as padded arrays; node 0 is each tree's root."""

    def __init__(self, data: dict):
        self._init = float(data["init"])
        self._rate = float(data["rate"])
        self._depth = int(data["depth"])
        self.feature = np.asarray(data["feature"], dtype=int)
        self._threshold = np.asarray(data["threshold"], dtype=float)
        self._left = np.asarray(data["left"], dtype=int)
        self._right = np.asarray(data["right"], dtype=int)
        self._value = np.asarray(data["value"], dtype=float)
        shapes = {a.shape for a in (self.feature, self._threshold, self._left, self._right, self._value)}  # fmt: skip
        if len(shapes) != 1 or self.feature.ndim != 2:
            raise ValueError("tree arrays must share one (trees, nodes) shape")

    def predict(self, X) -> np.ndarray:
        # scikit-learn compares float32 inputs with its float64 thresholds; do the same.
        X = np.asarray(X, dtype=np.float32)
        rows = np.arange(len(X))[:, None]
        trees = np.arange(self.feature.shape[0])
        node = np.zeros((len(X), len(trees)), dtype=int)
        for _ in range(self._depth):
            feature = self.feature[trees, node]
            leaf = feature < 0
            left = X[rows, np.where(leaf, 0, feature)] <= self._threshold[trees, node]
            below = np.where(left, self._left[trees, node], self._right[trees, node])
            node = np.where(leaf, node, below)
        return self._init + self._rate * self._value[trees, node].sum(axis=1)


class _Ridge:
    """Ridge regression on standardised inputs."""

    def __init__(self, data: dict):
        self._mean = np.asarray(data["mean"], dtype=float)
        self._scale = np.asarray(data["scale"], dtype=float)
        self._coef = np.asarray(data["coef"], dtype=float)
        self._intercept = float(data["intercept"])
        self.feature = np.arange(len(self._coef))

    def predict(self, X) -> np.ndarray:
        return ((np.asarray(X, dtype=float) - self._mean) / self._scale) @ self._coef + (
            self._intercept
        )


_KINDS = {"gbr": _Trees, "ridge": _Ridge}


@dataclass(frozen=True)
class RouterModel:
    policy: str
    feature_names: tuple
    actions: tuple
    costs: np.ndarray
    time_weight: float
    regressors: tuple

    @classmethod
    def from_dict(cls, data: dict) -> "RouterModel":
        if data.get("format") != FORMAT:
            raise ValueError(f"model format {data.get('format')!r}, expected {FORMAT}")
        if data["policy"] not in POLICIES:
            raise ValueError(f"unknown policy {data['policy']!r}")
        actions = tuple(Pipeline.from_dict(action) for action in data["actions"])
        regressors = tuple(_KINDS[r["kind"]](r) for r in data["regressors"])
        costs = np.asarray(data["costs"], dtype=float)
        names = tuple(data["feature_names"])
        expected = 0 if data["policy"] == "static" else len(actions)
        if not actions or len(regressors) != expected or costs.shape != (len(actions),):
            raise ValueError("actions, costs and regressors disagree")
        if any(r.feature.max() >= len(names) for r in regressors):
            raise ValueError("a regressor reads a feature the model does not name")
        return cls(data["policy"], names, actions, costs, float(data["time_weight"]), regressors)

    def predict_cer(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.clip(np.column_stack([r.predict(X) for r in self.regressors]), 0.0, 1.0)


def load_model(path=_MODEL_PATH) -> RouterModel:
    """Read a model file. Raises on a missing or malformed one."""
    return RouterModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class Router:
    """Loads the shipped model on first use and picks a pipeline per image."""

    def __init__(self, model_path=None, time_weight: float | None = None):
        self._path = Path(model_path) if model_path else _MODEL_PATH
        self._time_weight = time_weight
        self._model: RouterModel | None = None
        self._loaded = False
        self._warned = False

    def _load(self) -> RouterModel | None:
        if not self._loaded:
            try:
                self._model = load_model(self._path)
            except Exception as e:
                logger.warning(f"Pipeline routing is off, using {DEFAULT.name}: {e}")
            self._loaded = True
        return self._model

    @property
    def available(self) -> bool:
        """True when a model is loaded and it chooses between pipelines."""
        model = self._load()
        return model is not None and model.policy != "static"

    @property
    def policy(self) -> str:
        model = self._load()
        return model.policy if model else "static"

    @property
    def actions(self) -> tuple:
        model = self._load()
        return model.actions if model else (DEFAULT,)

    @property
    def default(self) -> Pipeline:
        return self.actions[0]

    def choose_vector(self, x) -> int:
        """Index of the action for a full model input (features, plus confidence stats in a cascade)."""
        model = self._load()
        if model is None or model.policy == "static":
            return 0
        x = np.asarray(x, dtype=float)
        if x.shape != (len(model.feature_names),) or not np.isfinite(x).all():
            if not self._warned:
                logger.warning(
                    f"Router expects {len(model.feature_names)} finite inputs, got {x.shape}: "
                    f"using {model.actions[0].name}"
                )
                self._warned = True
            return 0
        weight = model.time_weight if self._time_weight is None else self._time_weight
        return int(choose_actions(model.predict_cer(x), model.costs, weight)[0])

    def choose(self, features, first_conf: list[list[float]] | None = None) -> Pipeline:
        """Pipeline for an image's features; a cascade also takes the first pass's confidences."""
        x = np.asarray(features, dtype=float)
        if self.policy == "cascade":
            x = np.concatenate([x, conf_stats(first_conf)])
        return self.actions[self.choose_vector(x)]
