"""Select the pipelines, the model and the policy; export and evaluate the router.

1. Actions: greedy forward selection from the pool by the out-of-fold CER of the
   routed policy, on train + validation texts.
2. Model per policy by GroupKFold over those texts, minimising regret to the oracle.
3. Default time weight: the largest one that keeps out-of-fold CER within 0.005
   of the accuracy-only policy.
4. The policy with the lower out-of-fold CER ships; the model is written into
   the package. With a single selected action a static model ships.
5. Final numbers on test, unseen fonts, receipts and browser pages; the shipped
   policy is scored through sniptext.router.Router.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import GroupKFold

from benchmarks.evaluate import effective, paired, realized, summarize
from benchmarks.legacy import legacy_arrays, load_legacy
from benchmarks.metrics import normalize_text
from benchmarks.modeling import export_model, fit, predict, write_model
from benchmarks.run_eval import DEV_SPLITS, EVAL_SLICES
from sniptext.pipelines import LARGE_IMAGE, MAX_ROUTED_PIXELS, V04, Pipeline
from sniptext.router import Router, choose_actions

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results.json"
_TIMING = _HERE / "timing.json"
_LEGACY = _HERE / "legacy_v04.json"
_EVAL = _HERE / "router_eval.json"
_MODEL = _HERE.parent / "sniptext" / "data" / "router_model.json"

POLICIES = ("pre_ocr", "cascade")
CANDIDATES = [{"kind": "ridge", "alpha": alpha} for alpha in (0.1, 1.0, 10.0)] + [
    {"kind": "gbr", "n_estimators": n, "max_depth": depth, "learning_rate": rate}
    for n in (100, 300)
    for depth in (2, 3)
    for rate in (0.05, 0.1)
]
SELECTION_SPEC = {"kind": "gbr", "n_estimators": 100, "max_depth": 2, "learning_rate": 0.05}
MAX_ACTIONS = 4
MIN_GAIN = 0.003
TIME_WEIGHTS = [0.0] + [float(w) for w in np.geomspace(0.005, 2.0, 20)]
_TOLERANCE = 0.005


def load_arrays(results_path, timing_path) -> dict:
    """results.json and timing.json as column arrays; untimed rows have NaN seconds."""
    payload = json.loads(Path(results_path).read_text())
    timing = json.loads(Path(timing_path).read_text())
    rows = payload["rows"]
    names = [p["name"] for p in payload["pipelines"]]
    seconds = np.full((len(rows), len(names)), np.nan)
    feature_seconds = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        timed = timing["rows"].get(row["image"])
        if timed:
            seconds[i] = [timed["time"][name] for name in names]
            feature_seconds[i] = timed["features"]
    return {
        "rows": rows,
        "language": payload["language"],
        "feature_names": list(payload["feature_names"]),
        "conf_stat_names": list(payload["conf_stat_names"]),
        "pipelines": payload["pipelines"],
        "names": names,
        "split": np.array([row["split"] for row in rows]),
        "text_id": np.array([row["text_id"] for row in rows]),
        "image": np.array([row["image"] for row in rows]),
        "features": np.array([row["features"] for row in rows], dtype=float),
        "routable": np.array([row["pixels"] <= MAX_ROUTED_PIXELS for row in rows]),
        "cer": np.array([[row["cer"][n] for n in names] for row in rows], dtype=float),
        "empty": np.array([[not normalize_text(row["text"][n]) for n in names] for row in rows]),
        "conf_stats": np.array(
            [[row["conf_stats"][n] for n in names] for row in rows], dtype=float
        ),
        "seconds": seconds,
        "feature_seconds": feature_seconds,
        "loadavg": [timing["loadavg_start"], timing["loadavg_end"]],
    }


def action_costs(policy: str, mean_seconds) -> np.ndarray:
    """Seconds each action costs the policy. In a cascade the default has already run."""
    costs = np.asarray(mean_seconds, dtype=float).copy()
    if policy == "cascade":
        costs[1:] += costs[0]
    return costs


def design(data: dict, policy: str, chosen: list, mean_seconds):
    """Model inputs, delivered CER and seconds per chosen action, and the input names.

    *chosen* are pool columns, the default first. Rows without a timing use the
    pipelines' mean seconds.
    """
    measured = data["seconds"][:, chosen]
    seconds = np.where(np.isnan(measured), np.asarray(mean_seconds)[chosen], measured)
    cer, seconds = effective(policy, data["cer"][:, chosen], seconds, data["empty"][:, chosen])
    if policy == "cascade":
        X = np.hstack([data["features"], data["conf_stats"][:, chosen[0]]])
        names = data["feature_names"] + data["conf_stat_names"]
    else:
        X = data["features"]
        names = list(data["feature_names"])
    return X, cer, seconds, names


def oof_predictions(spec: dict, X, y, groups, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predicted CER; a text is never in both the fitting and the predicted fold."""
    pred = np.zeros((len(X), y.shape[1]))
    for fit_rows, held_out in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pred[held_out] = predict(fit(spec, X[fit_rows], y[fit_rows]), X[held_out])
    return pred


def _policy_cer(pred, y, costs, weight) -> float:
    return float(realized(y, choose_actions(pred, costs, weight)).mean())


def select_actions(
    data: dict,
    dev: np.ndarray,
    mean_seconds,
    spec: dict = SELECTION_SPEC,
    max_actions: int = MAX_ACTIONS,
    min_gain: float = MIN_GAIN,
):
    """Greedy forward selection of pool columns by the out-of-fold CER of the routed policy.

    Starts from the best static pipeline, which becomes the default. Returns the
    chosen columns and one record per step, including the rejected last one.
    """
    names = data["names"]
    groups = data["text_id"][dev]
    static = np.minimum(data["cer"][dev], 1.0).mean(axis=0)
    chosen = [int(static.argmin())]
    current = float(static[chosen[0]])
    steps = [{"added": names[chosen[0]], "oof_cer": current, "oracle": current}]
    while len(chosen) < max_actions:
        trials = {}
        for candidate in range(len(names)):
            if candidate in chosen:
                continue
            X, cer, _, _ = design(data, "pre_ocr", chosen + [candidate], mean_seconds)
            y = np.minimum(cer[dev], 1.0)
            pred = oof_predictions(spec, X[dev], y, groups)
            trials[candidate] = (
                _policy_cer(pred, y, np.zeros(y.shape[1]), 0.0),
                float(y.min(axis=1).mean()),
            )
        best = min(trials, key=lambda candidate: trials[candidate][0])
        oof_cer, oracle = trials[best]
        if current - oof_cer < min_gain:
            steps.append(
                {"added": None, "rejected": names[best], "oof_cer": oof_cer, "oracle": oracle}
            )
            break
        chosen.append(best)
        current = oof_cer
        steps.append({"added": names[best], "oof_cer": oof_cer, "oracle": oracle})
    return chosen, steps


def select_model(X, y, groups, costs, candidates):
    """Candidate with the lowest out-of-fold regret at time weight 0."""
    table, best = [], None
    oracle = float(y.min(axis=1).mean())
    for spec in candidates:
        pred = oof_predictions(spec, X, y, groups)
        cer = _policy_cer(pred, y, costs, 0.0)
        table.append({"spec": spec, "oof_cer": cer, "oof_regret": cer - oracle})
        if np.isfinite(cer) and (best is None or cer < best[2]):
            best = (spec, pred, cer)
    if best is None:
        raise ValueError("no model candidate produced a finite out-of-fold CER")
    return best[0], best[1], table


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


def _importance(models, X, y, costs, weight, names, repeats: int = 20, seed: int = 0) -> dict:
    """Mean CER change on held-out rows when one input is shuffled."""
    rng = np.random.default_rng(seed)
    base = _policy_cer(predict(models, X), y, costs, weight)
    out = {}
    for k, name in enumerate(names):
        deltas = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, k] = rng.permutation(shuffled[:, k])
            deltas.append(_policy_cer(predict(models, shuffled), y, costs, weight) - base)
        out[name] = float(np.mean(deltas))
    return out


def _large_column(names: list, default: int) -> int:
    """Pool column of the pipeline the app runs on an image above its size limit."""
    return names.index(LARGE_IMAGE.name) if LARGE_IMAGE.name in names else default


def delivered(data: dict, router: Router, mask: np.ndarray) -> np.ndarray:
    """Pool column whose text the app returns for each masked row.

    The app does not route an image above its size limit: it runs LARGE_IMAGE.
    """
    columns = [data["names"].index(p.name) for p in router.actions]
    X = data["features"][mask]
    if router.policy == "cascade":
        X = np.hstack([X, data["conf_stats"][mask][:, columns[0]]])
    empty = data["empty"][mask]
    routable = data["routable"][mask]
    large = _large_column(data["names"], columns[0])
    out = np.zeros(len(X), dtype=int)
    for i, x in enumerate(X):
        if not routable[i]:
            out[i] = large
            continue
        column = columns[router.choose_vector(x)]
        out[i] = columns[0] if empty[i, column] else column
    return out


def _degradation_kind(row: dict) -> str:
    return "two combined" if "+" in row["degradation"] else row["degradation"]


def _dataset_card(data: dict) -> dict:
    rows = data["rows"]
    synthetic = [row for row in rows if row["source"] == "synthetic"]
    splits = [row["split"] for row in rows]
    return {
        "n": len(rows),
        "texts": len({row["text_id"] for row in synthetic}),
        "language": data["language"],
        "by_split": {s: splits.count(s) for s in dict.fromkeys(splits)},
        "by_lang": {v: sum(r["lang"] == v for r in synthetic) for v in ("en", "ru")},
        "by_content": {
            v: sum(r["content"] == v for r in synthetic) for v in ("prose", "code", "ui")
        },
        "fonts": sorted({row["font"] for row in synthetic}),
        "unrouted": int((~data["routable"]).sum()),
    }


def _groups(rows: list, per_image: dict, shown: tuple, key) -> dict:
    values = np.array([key(row) for row in rows])
    return {
        str(group): {
            "n": int((values == group).sum()),
            **{label: float(per_image[label][values == group].mean()) for label in shown},
        }
        for group in dict.fromkeys(values.tolist())
    }


def run(results, timing, legacy_path, model_path, eval_path, candidates=CANDIDATES) -> dict:
    data = load_arrays(results, timing)
    names = data["names"]
    dev = np.isin(data["split"], DEV_SPLITS)
    groups = data["text_id"][dev]
    mean_seconds = np.nanmean(data["seconds"][dev], axis=0)
    clipped_pool = np.minimum(data["cer"], 1.0)

    chosen, steps = select_actions(data, dev, mean_seconds)
    best_static = chosen[0]
    large_column = _large_column(names, best_static)
    pipelines = [Pipeline.from_dict(data["pipelines"][k]) for k in chosen]

    fitted = {}
    if len(chosen) > 1:
        for policy in POLICIES:
            X, cer, seconds, input_names = design(data, policy, chosen, mean_seconds)
            y = np.minimum(cer, 1.0)
            costs = action_costs(policy, mean_seconds[chosen])
            spec, pred, selection = select_model(X[dev], y[dev], groups, costs, candidates)
            weight = pick_time_weight(pred, y[dev], costs)
            actions = choose_actions(pred, costs, weight)
            fitted[policy] = {
                "X": X, "cer": cer, "y": y, "seconds": seconds, "names": input_names,
                "costs": costs, "spec": spec, "weight": weight, "selection": selection,
                "oof": {"cer": float(realized(y[dev], actions).mean()),
                        "time": float(costs[actions].mean())},
                "ablation": _ablation(spec, X[dev], y[dev], groups, costs, input_names,
                                      _policy_cer(pred, y[dev], costs, 0.0)),
                "models": fit(spec, X[dev], y[dev]),
            }  # fmt: skip
        shipped = select_policy({policy: fitted[policy]["oof"] for policy in POLICIES})
        ship = fitted[shipped]
        model = export_model(shipped, ship["names"], pipelines, ship["costs"], ship["weight"],
                             ship["spec"], ship["models"])  # fmt: skip
    else:
        shipped = "static"
        model = export_model("static", data["feature_names"], pipelines,
                             [mean_seconds[best_static]], 0.0, None, [])  # fmt: skip
    write_model(model_path, model)
    router = Router(model_path)

    legacy = load_legacy(legacy_path)
    shipped_label = f"always_{names[best_static]}" if shipped == "static" else f"router_{shipped}"
    slices, per_image, v04_delta = {}, {}, {}
    for name in EVAL_SLICES:
        mask = data["split"] == name
        if not mask.any():
            continue
        if np.isnan(data["seconds"][mask]).any():
            raise ValueError(f"slice {name!r} has images without a timing")
        clusters = data["text_id"][mask]
        pool_cer, pool_seconds = data["cer"][mask], data["seconds"][mask]
        clipped = clipped_pool[mask]
        oracle_shipped = clipped[:, chosen].min(axis=1)
        reference = clipped[:, best_static]
        summaries, vectors = [], {}

        def add(label, cer, seconds, **kwargs):
            summaries.append(summarize(label, cer, seconds, clusters, **kwargs))
            vectors[label] = np.minimum(cer, 1.0)

        for policy, f in fitted.items():
            X = f["X"][mask]
            if policy == shipped:
                actions = np.array([router.choose_vector(x) for x in X])
            else:
                actions = choose_actions(predict(f["models"], X), f["costs"], f["weight"])
            # The app does not route an image above its size limit.
            large = ~data["routable"][mask]
            share = {
                names[k]: float(np.mean((actions == i) & ~large)) for i, k in enumerate(chosen)
            }
            if large.any():
                share[names[large_column]] = share.get(names[large_column], 0.0) + float(
                    large.mean()
                )
            add(f"router_{policy}",
                np.where(large, pool_cer[:, large_column], realized(f["cer"][mask], actions)),
                np.where(large, pool_seconds[:, large_column], realized(f["seconds"][mask], actions)),
                oracle=oracle_shipped, reference=reference, share=share)  # fmt: skip
        static_columns = list(chosen)
        if V04.name in names and names.index(V04.name) not in static_columns:
            static_columns.append(names.index(V04.name))
        for k in static_columns:
            add(f"always_{names[k]}", pool_cer[:, k], pool_seconds[:, k], oracle=oracle_shipped,
                reference=None if k == best_static else reference)  # fmt: skip
        old = legacy_arrays(legacy, data["image"][mask])
        if old["present"].all():
            add("router_v04", realized(old["cer"], old["action"]), old["seconds"],
                reference=reference)  # fmt: skip
            v04_delta[name] = paired(vectors[shipped_label], vectors["router_v04"], clusters)
        for label, columns in (("oracle", chosen), ("oracle_pool", list(range(len(names))))):
            best = np.array(columns)[clipped[:, columns].argmin(axis=1)]
            add(label, realized(pool_cer, best), realized(pool_seconds, best), reference=reference)
        slices[name] = summaries
        per_image[name] = vectors

    test_delta = next(s for s in slices["test"] if s["policy"] == shipped_label).get("delta")
    browser = slices.get("browser")
    browser_delta = (
        next(s for s in browser if s["policy"] == shipped_label).get("delta") if browser else None
    )
    criteria = {
        "router_beats_best_static": bool(test_delta is not None and test_delta[2] < 0),
        "not_worse_than_v04": bool("test" in v04_delta and v04_delta["test"][1] <= 0),
        "browser_not_worse": bool(browser_delta is None or browser_delta[1] <= 0),
        "test_delta": test_delta,
        "v04_delta": v04_delta.get("test"),
        "browser_delta": browser_delta,
    }

    wanted = (f"always_{names[best_static]}", f"always_{V04.name}", shipped_label, "oracle")
    shown = tuple(label for label in dict.fromkeys(wanted) if label in per_image["test"])
    breakdown = {}
    for name, keys in (
        ("test", (("degradation", _degradation_kind), ("theme", lambda r: r["theme"]),
                  ("lang", lambda r: r["lang"]), ("content", lambda r: r["content"]))),
        ("browser", (("scale", lambda r: f"{r['scale']:g}x"), ("theme", lambda r: r["theme"]),
                     ("content", lambda r: r["content"]))),
    ):  # fmt: skip
        if name in per_image:
            rows = [row for row, keep in zip(data["rows"], data["split"] == name) if keep]
            breakdown[name] = {
                label: _groups(rows, per_image[name], shown, key) for label, key in keys
            }
    test_rows = [row for row, keep in zip(data["rows"], data["split"] == "test") if keep]
    clean_vs_degraded = _groups(
        test_rows, per_image["test"], shown,
        lambda row: "clean" if row["degradation"] == "none" else "degraded",
    )  # fmt: skip

    old_dev = legacy_arrays(legacy, data["image"][dev])
    known = old_dev["present"]
    ours = clipped_pool[dev][known][:, chosen].min(axis=1)
    old_cer = np.minimum(old_dev["cer"][known], 1.0)
    easyocr = {
        "n": int(known.sum()),
        "shipped": float(ours.mean()),
        "with_easyocr": float(np.minimum(ours, old_cer[:, 1]).mean()),
        "v04_actions": float(old_cer.min(axis=1).mean()),
    }

    test = data["split"] == "test"
    policies = {}
    for policy, f in fitted.items():
        pred_test = predict(f["models"], f["X"][test])
        curve = []
        for weight in TIME_WEIGHTS:
            actions = choose_actions(pred_test, f["costs"], weight)
            curve.append([weight, float(realized(f["y"][test], actions).mean()),
                          float(realized(f["seconds"][test], actions).mean())])  # fmt: skip
        policies[policy] = {
            "spec": f["spec"],
            "time_weight": f["weight"],
            "feature_names": f["names"],
            "costs": [float(c) for c in f["costs"]],
            "oof": f["oof"],
            "selection": f["selection"],
            "ablation": f["ablation"],
            "importance": _importance(
                f["models"], f["X"][test], f["y"][test], f["costs"], f["weight"], f["names"]
            ),  # fmt: skip
            "curve": curve,
        }

    evaluation = {
        "dataset": _dataset_card(data),
        "pool": data["pipelines"],
        "pool_dev_cer": {n: float(v) for n, v in zip(names, clipped_pool[dev].mean(axis=0))},
        "pool_seconds": {n: float(v) for n, v in zip(names, mean_seconds)},
        "action_selection": steps,
        "actions": [names[k] for k in chosen],
        "shipped": shipped,
        "best_static": names[best_static],
        "policies": policies,
        "slices": slices,
        "per_slice_delta_v04": v04_delta,
        "breakdown": breakdown,
        "clean_vs_degraded": clean_vs_degraded,
        "easyocr": easyocr,
        "criteria": criteria,
        "feature_time_mean": float(np.nanmean(data["feature_seconds"])),
        "loadavg": data["loadavg"],
    }
    Path(eval_path).write_text(json.dumps(evaluation, indent=1))
    return evaluation


def main() -> None:
    ev = run(_RESULTS, _TIMING, _LEGACY, _MODEL, _EVAL)
    print(f"actions: {ev['actions']}  shipped: {ev['shipped']}  best static: {ev['best_static']}")
    for step in ev["action_selection"]:
        print(
            f"  {step.get('added') or 'stop at ' + step['rejected']:28s} oof {step['oof_cer']:.4f}"
        )
    for name, summaries in ev["slices"].items():
        print(name)
        for s in summaries:
            mean, low, high = s["cer"]
            delta = (
                "" if "delta" not in s else "  delta {:+.3f} [{:+.3f}, {:+.3f}]".format(*s["delta"])
            )
            print(f"  {s['policy']:30s} CER {mean:.3f} [{low:.3f}, {high:.3f}] "
                  f"{s['time'] * 1000:4.0f} ms{delta}")  # fmt: skip
    print("criteria:", json.dumps(ev["criteria"]))


if __name__ == "__main__":
    main()
