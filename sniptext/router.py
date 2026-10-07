"""Cost-aware choice between OCR actions.

The policy is: predict the character error rate of each action for this image,
then take the action minimising predicted CER + time_weight * seconds. The
regressors are fitted from a table shipped with the package, so the model the
app runs is the one the benchmark evaluates.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

ACTIONS = ("tesseract", "easyocr", "merge")
CONF_STAT_NAMES = (
    "tess_conf_mean",
    "tess_conf_min",
    "tess_conf_p10",
    "tess_low_share",
    "tess_n_words",
)
_LOW_CONFIDENCE = 0.6
_TARGET_COLUMNS = tuple(f"cer_{action}" for action in ACTIONS)
_TABLE_PATH = Path(__file__).resolve().parent / "data" / "router_train.csv.gz"


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


def action_costs(policy: str, t_tesseract: float, t_easyocr: float) -> np.ndarray:
    """Seconds each action costs. In a cascade Tesseract has already run."""
    both = t_tesseract + t_easyocr
    if policy == "pre_ocr":
        return np.array([t_tesseract, t_easyocr, both])
    if policy == "cascade":
        return np.array([t_tesseract, both, both])
    raise ValueError(f"unknown policy {policy!r}")


def choose_actions(pred_cer: np.ndarray, costs: np.ndarray, time_weight: float) -> np.ndarray:
    """Per row, the index of the action minimising predicted CER + time_weight * cost.

    Ties go to the cheaper action, then to the earlier one in ACTIONS.
    """
    costs = np.asarray(costs, dtype=float)
    objective = np.asarray(pred_cer, dtype=float) + time_weight * costs
    tied = objective <= objective.min(axis=1, keepdims=True) + 1e-9
    return np.where(tied, np.broadcast_to(costs, objective.shape), np.inf).argmin(axis=1)


@dataclass(frozen=True)
class RouterTable:
    feature_names: tuple
    X: np.ndarray
    y: np.ndarray
    meta: dict
    digest: str


def write_table(path, feature_names, X, y, meta: dict) -> None:
    """Write features, per-action clipped CER and metadata as a gzipped CSV."""
    out = io.StringIO()
    out.write("# " + json.dumps(meta, sort_keys=True) + "\n")
    out.write(",".join((*feature_names, *_TARGET_COLUMNS)) + "\n")
    for features, targets in zip(np.asarray(X), np.asarray(y)):
        out.write(",".join(f"{v:.6g}" for v in (*features, *targets)) + "\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the file byte-identical across rebuilds.
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(out.getvalue().encode("utf-8"))


def read_table(path) -> RouterTable:
    """Read a table written by write_table. Raises OSError or ValueError."""
    try:
        raw = gzip.decompress(Path(path).read_bytes())
    except (gzip.BadGzipFile, EOFError) as e:
        raise ValueError(f"{path}: not a gzip file ({e})") from e
    lines = raw.decode("utf-8").splitlines()
    if len(lines) < 3 or not lines[0].startswith("# "):
        raise ValueError(f"{path}: not a router table")
    meta = json.loads(lines[0][2:])
    columns = lines[1].split(",")
    if tuple(columns[-len(ACTIONS) :]) != _TARGET_COLUMNS:
        raise ValueError(f"{path}: unexpected target columns {columns[-len(ACTIONS) :]}")
    data = np.array([[float(v) for v in line.split(",")] for line in lines[2:]], dtype=float)
    n_features = len(columns) - len(ACTIONS)
    return RouterTable(
        feature_names=tuple(columns[:n_features]),
        X=data[:, :n_features],
        y=data[:, n_features:],
        meta=meta,
        digest=hashlib.sha256(raw).hexdigest(),
    )


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


class RouterModel:
    """One CER regressor per action."""

    def __init__(self, spec: dict):
        self.spec = spec
        self._models: list = []

    def fit(self, X, y) -> "RouterModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._models = [build_regressor(self.spec).fit(X, y[:, k]) for k in range(len(ACTIONS))]
        return self

    def predict_cer(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.clip(np.column_stack([m.predict(X) for m in self._models]), 0.0, 1.0)


class Router:
    """Loads the shipped table lazily, fits or reuses a cached model, picks an action."""

    def __init__(self, table_path=None, cache_dir=None, time_weight: float | None = None):
        self._table_path = Path(table_path) if table_path else _TABLE_PATH
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "sniptext"
        self._time_weight = time_weight
        self._table: RouterTable | None = None
        self._model: RouterModel | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import sklearn
        except ImportError:
            logger.warning("scikit-learn is not installed: engine routing is off, using Tesseract")
            return
        try:
            table = read_table(self._table_path)
            model = self._cached_model(table, sklearn.__version__)
        except Exception as e:
            logger.warning(f"Engine routing is off, using Tesseract: {e}")
            return
        self._table, self._model = table, model

    def _cached_model(self, table: RouterTable, sklearn_version: str) -> RouterModel:
        cache = self._cache_dir / f"router-{table.digest[:16]}-sk{sklearn_version}.pkl"
        try:
            with open(cache, "rb") as f:
                model = pickle.load(f)
            if isinstance(model, RouterModel):
                return model
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"Ignoring unreadable router cache {cache}: {e}")
        model = RouterModel(table.meta["model"]).fit(table.X, table.y)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(model, f)
        except OSError as e:
            logger.debug(f"Could not write router cache {cache}: {e}")
        return model

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    @property
    def policy(self) -> str:
        self._load()
        return self._table.meta["policy"] if self._table else "pre_ocr"

    @property
    def tesseract_call(self) -> str:
        self._load()
        return self._table.meta.get("tesseract_call", "detailed") if self._table else "plain"

    def choose_vector(self, x) -> str:
        """Action for a full model input (image features, plus confidence stats in a cascade)."""
        if not self.available:
            return ACTIONS[0]
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self._table.feature_names),):
            logger.warning(
                f"Router expects {len(self._table.feature_names)} features, got {x.shape}: "
                "using Tesseract"
            )
            return ACTIONS[0]
        meta = self._table.meta
        costs = action_costs(meta["policy"], meta["time"]["tesseract"], meta["time"]["easyocr"])
        weight = meta["time_weight"] if self._time_weight is None else self._time_weight
        return ACTIONS[int(choose_actions(self._model.predict_cer(x), costs, weight)[0])]

    def choose(self, features, tesseract_conf: list[list[float]] | None = None) -> str:
        """Action for an image's features; a cascade also takes Tesseract's word confidences."""
        if not self.available:
            return ACTIONS[0]
        x = np.asarray(features, dtype=float)
        if self._table.meta["policy"] == "cascade":
            x = np.concatenate([x, conf_stats(tesseract_conf)])
        return self.choose_vector(x)
