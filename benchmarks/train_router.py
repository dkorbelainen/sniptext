"""Select, evaluate and export the engine router.

1. Model selection per policy by GroupKFold over train + validation texts,
   minimising regret to the oracle.
2. Default time weight: the largest one that keeps out-of-fold CER within 0.005
   of the accuracy-only policy.
3. The policy with the lower out-of-fold CER ships; the training table is written
   into the package.
4. Final numbers on test, the unseen-font slice and out-of-domain receipts; the
   shipped policy is scored through sniptext.router.Router.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import GroupKFold

from benchmarks.evaluate import realized, summarize, time_matrix, weak_label_agreement
from benchmarks.legacy_policy import legacy_actions
from sniptext.router import (
    ACTIONS,
    Router,
    RouterModel,
    action_costs,
    choose_actions,
    read_table,
    write_table,
)

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results.json"
_EVAL = _HERE / "router_eval.json"
_TIMING_CPU = _HERE / "timing_cpu.json"
_TABLE = _HERE.parent / "sniptext" / "data" / "router_train.csv.gz"

POLICIES = ("pre_ocr", "cascade")
CANDIDATES = [{"kind": "ridge", "alpha": alpha} for alpha in (0.1, 1.0, 10.0)] + [
    {"kind": "gbr", "n_estimators": n, "max_depth": depth, "learning_rate": rate}
    for n in (100, 300)
    for depth in (2, 3)
    for rate in (0.05, 0.1)
]
TIME_WEIGHTS = [0.0] + [float(w) for w in np.geomspace(0.005, 2.0, 20)]
_TOLERANCE = 0.005
_DEV_SPLITS = ("train", "val")
_EVAL_SLICES = ("test", "unseen_font", "ood")
_CER_KEYS = ("tesseract_plain", "tesseract", "easyocr", "merge")
_TIME_KEYS = ("tesseract_plain", "tesseract", "easyocr")


def load_arrays(path) -> dict:
    """results.json as column arrays."""
    payload = json.loads(Path(path).read_text())
    rows = payload["rows"]

    def column(key):
        return np.array([row[key] for row in rows])

    return {
        "rows": rows,
        "language": payload["language"],
        "feature_names": list(payload["feature_names"]),
        "conf_stat_names": list(payload["conf_stat_names"]),
        "split": column("split"),
        "text_id": column("text_id"),
        "features": column("features").astype(float),
        "conf_stats": column("conf_stats").astype(float),
        "legacy_features": column("legacy_features").astype(float),
        "cer": {k: np.array([row["cer"][k] for row in rows], dtype=float) for k in _CER_KEYS},
        "time": {k: np.array([row["time"][k] for row in rows], dtype=float) for k in _TIME_KEYS},
    }


def _tesseract_gap(data: dict, mask: np.ndarray) -> float:
    detailed = np.minimum(data["cer"]["tesseract"][mask], 1.0).mean()
    plain = np.minimum(data["cer"]["tesseract_plain"][mask], 1.0).mean()
    return float(detailed - plain)


def tesseract_call_rule(data: dict, mask: np.ndarray) -> str:
    """'detailed' unless image_to_data text is worse than image_to_string by more than 0.01."""
    return "detailed" if _tesseract_gap(data, mask) <= 0.01 else "plain"


def design(data: dict, policy: str, tesseract_call: str):
    """Model inputs, raw per-action CER, per-action seconds and input names for a policy."""
    # A cascade needs Tesseract's confidences, so it always uses the detailed call.
    key = "tesseract" if policy == "cascade" or tesseract_call == "detailed" else "tesseract_plain"
    cer = np.column_stack([data["cer"][key], data["cer"]["easyocr"], data["cer"]["merge"]])
    times = time_matrix(policy, data["time"][key], data["time"]["easyocr"])
    if policy == "cascade":
        X = np.hstack([data["features"], data["conf_stats"]])
        names = data["feature_names"] + data["conf_stat_names"]
    else:
        X = data["features"]
        names = list(data["feature_names"])
    return X, cer, times, names


def oof_predictions(spec: dict, X, y, groups, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predicted CER; a text is never in both the fitting and the predicted fold."""
    pred = np.zeros((len(X), y.shape[1]))
    for fit_rows, held_out in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pred[held_out] = RouterModel(spec).fit(X[fit_rows], y[fit_rows]).predict_cer(X[held_out])
    return pred


def _policy_cer(pred, y, costs, weight) -> float:
    return float(realized(y, choose_actions(pred, costs, weight)).mean())


def select_model(X, y, groups, costs, candidates):
    """Candidate with the lowest out-of-fold regret at time weight 0."""
    table, best_pred, best_regret = [], None, np.inf
    oracle = float(y.min(axis=1).mean())
    for spec in candidates:
        pred = oof_predictions(spec, X, y, groups)
        cer = _policy_cer(pred, y, costs, 0.0)
        table.append({"spec": spec, "oof_cer": cer, "oof_regret": cer - oracle})
        if cer - oracle < best_regret:
            best_regret, best_pred, best_spec = cer - oracle, pred, spec
    return best_spec, best_pred, table


def pick_time_weight(pred, y, costs, weights=TIME_WEIGHTS, tolerance: float = _TOLERANCE) -> float:
    """Largest time weight whose CER stays within *tolerance* of the accuracy-only policy."""
    base = _policy_cer(pred, y, costs, 0.0)
    allowed = [w for w in weights if _policy_cer(pred, y, costs, w) <= base + tolerance]
    return float(max(allowed))


def select_policy(stats: dict) -> str:
    """Lower out-of-fold CER wins; within the tolerance, the faster policy."""
    first, second = (stats[p] for p in POLICIES)
    if abs(first["cer"] - second["cer"]) <= _TOLERANCE:
        return POLICIES[0] if first["time"] <= second["time"] else POLICIES[1]
    return POLICIES[0] if first["cer"] < second["cer"] else POLICIES[1]


def _ablation(spec, X, y, groups, costs, names, base_cer: float) -> dict:
    """Out-of-fold CER change when one input is removed (positive: the input helps)."""
    out = {}
    for k, name in enumerate(names):
        pred = oof_predictions(spec, np.delete(X, k, axis=1), y, groups)
        out[name] = _policy_cer(pred, y, costs, 0.0) - base_cer
    return out


def _importance(model, X, y, costs, weight, names, repeats: int = 20, seed: int = 0) -> dict:
    """Mean CER change on held-out rows when one input is shuffled."""
    rng = np.random.default_rng(seed)
    base = _policy_cer(model.predict_cer(X), y, costs, weight)
    out = {}
    for k, name in enumerate(names):
        deltas = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, k] = rng.permutation(shuffled[:, k])
            deltas.append(_policy_cer(model.predict_cer(shuffled), y, costs, weight) - base)
        out[name] = float(np.mean(deltas))
    return out


def _degradation_kind(row: dict) -> str:
    return "two combined" if "+" in row["degradation"] else row["degradation"]


def _dataset_card(data: dict) -> dict:
    def count(key):
        values = [key(row) for row in data["rows"]]
        return {v: values.count(v) for v in dict.fromkeys(values)}

    synthetic = [row for row in data["rows"] if row["source"] == "synthetic"]
    return {
        "n": len(data["rows"]),
        "texts": len({row["text_id"] for row in synthetic}),
        "language": data["language"],
        "by_split": count(lambda row: row["split"]),
        "by_lang": {v: sum(r["lang"] == v for r in synthetic) for v in ("en", "ru")},
        "by_content": {
            v: sum(r["content"] == v for r in synthetic) for v in ("prose", "code", "ui")
        },
        "fonts": sorted({row["font"] for row in synthetic}),
    }


def run(results, table_path, eval_path, timing_path, candidates=CANDIDATES) -> dict:
    data = load_arrays(results)
    dev = np.isin(data["split"], _DEV_SPLITS)
    groups = data["text_id"][dev]
    call = tesseract_call_rule(data, dev)
    t_easy = float(data["time"]["easyocr"][dev].mean())

    fitted = {}
    for policy in POLICIES:
        X, cer, times, names = design(data, policy, call)
        y = np.minimum(cer, 1.0)
        t_tess = float(times[dev][:, 0].mean())
        costs = action_costs(policy, t_tess, t_easy)
        spec, pred, selection = select_model(X[dev], y[dev], groups, costs, candidates)
        weight = pick_time_weight(pred, y[dev], costs)
        actions = choose_actions(pred, costs, weight)
        fitted[policy] = {
            "X": X, "cer": cer, "y": y, "times": times, "names": names, "costs": costs,
            "t_tess": t_tess, "spec": spec, "oof_pred": pred, "weight": weight,
            "selection": selection,
            "oof": {
                "cer": float(realized(y[dev], actions).mean()),
                "time": float(realized(times[dev], actions).mean()),
            },
            "ablation": _ablation(spec, X[dev], y[dev], groups, costs, names,
                                  _policy_cer(pred, y[dev], costs, 0.0)),
        }  # fmt: skip

    shipped = select_policy({policy: fitted[policy]["oof"] for policy in POLICIES})
    ship = fitted[shipped]
    write_table(
        table_path,
        ship["names"],
        ship["X"][dev],
        ship["y"][dev],
        {
            "policy": shipped,
            "model": ship["spec"],
            "time": {"tesseract": ship["t_tess"], "easyocr": t_easy},
            "time_weight": ship["weight"],
            "tesseract_call": "detailed" if shipped == "cascade" else call,
        },
    )
    # Fit from the written table, as the app does, so both see the same rounding.
    table = read_table(table_path)
    for policy in POLICIES:
        f = fitted[policy]
        if policy == shipped:
            f["model"] = RouterModel(f["spec"]).fit(table.X, table.y)
        else:
            f["model"] = RouterModel(f["spec"]).fit(f["X"][dev], f["y"][dev])

    base = fitted["pre_ocr"]
    best_static = int(np.argmin(base["y"][dev].mean(axis=0)))
    slices, per_image = {}, {}
    with tempfile.TemporaryDirectory() as cache:
        router = Router(table_path, cache_dir=cache)
        for name in _EVAL_SLICES:
            mask = data["split"] == name
            if not mask.any():
                continue
            clusters = data["text_id"][mask]
            n = int(mask.sum())
            cer, times = base["cer"][mask], base["times"][mask]
            reference = np.minimum(cer[:, best_static], 1.0)
            chosen = {
                f"always_{action}": (cer, np.full(n, k), times) for k, action in enumerate(ACTIONS)
            }
            chosen["legacy_rules"] = (cer, legacy_actions(data["legacy_features"][mask]), times)
            for policy in POLICIES:
                f = fitted[policy]
                if policy == shipped:
                    actions = np.array(
                        [ACTIONS.index(router.choose_vector(x)) for x in f["X"][mask]]
                    )
                else:
                    actions = choose_actions(
                        f["model"].predict_cer(f["X"][mask]), f["costs"], f["weight"]
                    )
                chosen[f"router_{policy}"] = (f["cer"][mask], actions, f["times"][mask])
            chosen["oracle"] = (cer, np.minimum(cer, 1.0).argmin(axis=1), times)
            slices[name] = [
                summarize(
                    label, c, a, t, clusters, None if label.startswith("always_") else reference
                )
                for label, (c, a, t) in chosen.items()
            ]
            per_image[name] = {
                label: realized(np.minimum(c, 1.0), a) for label, (c, a, _) in chosen.items()
            }

    test = data["split"] == "test"
    test_rows = [row for row, keep in zip(data["rows"], test) if keep]
    shown = ("always_tesseract", "always_easyocr", "always_merge", f"router_{shipped}", "oracle")
    breakdown = {}
    for label, key in (
        ("degradation", _degradation_kind),
        ("theme", lambda row: row["theme"]),
        ("lang", lambda row: row["lang"]),
        ("content", lambda row: row["content"]),
    ):
        values = np.array([key(row) for row in test_rows])
        breakdown[label] = {
            str(group): {
                "n": int((values == group).sum()),
                **{p: float(per_image["test"][p][values == group].mean()) for p in shown},
            }
            for group in dict.fromkeys(values.tolist())
        }

    policies = {}
    for policy in POLICIES:
        f = fitted[policy]
        pred_test = f["model"].predict_cer(f["X"][test])
        curve = []
        for weight in TIME_WEIGHTS:
            actions = choose_actions(pred_test, f["costs"], weight)
            curve.append(
                [
                    weight,
                    float(realized(f["y"][test], actions).mean()),
                    float(realized(f["times"][test], actions).mean()),
                ]
            )
        policies[policy] = {
            "spec": f["spec"],
            "time_weight": f["weight"],
            "feature_names": f["names"],
            "costs": [float(c) for c in f["costs"]],
            "oof": f["oof"],
            "selection": f["selection"],
            "ablation": f["ablation"],
            "importance": _importance(
                f["model"], f["X"][test], f["y"][test], f["costs"], f["weight"], f["names"]
            ),
            "curve": curve,
        }

    cpu = None
    if Path(timing_path).exists():
        timing = json.loads(Path(timing_path).read_text())
        ratio = timing["easyocr_cpu_mean"] / timing["easyocr_gpu_mean"]
        cpu = {"ratio": float(ratio), "always": {}, "policies": {}}
        cpu_times = time_matrix(
            "pre_ocr", base["times"][test][:, 0], data["time"]["easyocr"][test] * ratio
        )
        for k, action in enumerate(ACTIONS):
            cpu["always"][action] = {
                "cer": float(base["y"][test][:, k].mean()),
                "time": float(cpu_times[:, k].mean()),
            }
        for policy in POLICIES:
            f = fitted[policy]
            costs = action_costs(policy, f["t_tess"], t_easy * ratio)
            weight = pick_time_weight(f["oof_pred"], f["y"][dev], costs)
            actions = choose_actions(f["model"].predict_cer(f["X"][test]), costs, weight)
            times = time_matrix(
                policy, f["times"][test][:, 0], data["time"]["easyocr"][test] * ratio
            )
            cpu["policies"][policy] = {
                "time_weight": weight,
                "cer": float(realized(f["y"][test], actions).mean()),
                "time": float(realized(times, actions).mean()),
                "share": {a: float(np.mean(actions == k)) for k, a in enumerate(ACTIONS)},
            }

    synthetic = [row for row in data["rows"] if row["source"] == "synthetic"]
    evaluation = {
        "dataset": _dataset_card(data),
        "tesseract_call": call,
        "tesseract_call_gap": _tesseract_gap(data, dev),
        "shipped": shipped,
        "best_static": ACTIONS[best_static],
        "policies": policies,
        "slices": slices,
        "breakdown": breakdown,
        "cpu": cpu,
        "feature_time_mean": float(np.mean([row["time"]["features"] for row in synthetic])),
        "weak_labels": {
            "synthetic": weak_label_agreement(synthetic),
            "all": weak_label_agreement(data["rows"]),
        },
    }
    Path(eval_path).write_text(json.dumps(evaluation, indent=1))
    return evaluation


def main():
    ev = run(_RESULTS, _TABLE, _EVAL, _TIMING_CPU)
    ship = ev["policies"][ev["shipped"]]
    print(f"shipped: {ev['shipped']} {ship['spec']} time_weight={ship['time_weight']:.3f}")
    for summary in ev["slices"]["test"]:
        mean, low, high = summary["cer"]
        print(
            f"{summary['policy']:18s} CER {mean:.3f} [{low:.3f}, {high:.3f}] "
            f"time {summary['time'] * 1000:.0f} ms"
        )
    print("weak labels:", ev["weak_labels"]["synthetic"])


if __name__ == "__main__":
    main()
