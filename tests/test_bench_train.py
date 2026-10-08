import json

import numpy as np
import pytest

from benchmarks import train_router as tr
from sniptext.analyzer import FEATURE_NAMES
from sniptext.pipelines import MAX_ROUTED_PIXELS, V04
from sniptext.router import CONF_STAT_NAMES, Router, load_model

NAMES = ("p_base", "p_alt", "p_same", "p_bad")
SPLITS = (
    "train",
    "train",
    "train",
    "train",
    "val",
    "val",
    "test",
    "test",
    "unseen_font",
    "browser",
)
FAST = [{"kind": "gbr", "n_estimators": 40, "max_depth": 2, "learning_rate": 0.1}]
NOISE = FEATURE_NAMES.index("noise_level")


def fake_run(tmp_path, n_texts=100, seed=0, helpful=True, empty_alt=False):
    """p_alt is the better pipeline exactly on noisy images; p_same copies p_base."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, timing, legacy = [], {}, {}
    for t in range(n_texts):
        split = SPLITS[t % len(SPLITS)]
        for r in range(2):
            features = rng.random(len(FEATURE_NAMES))
            noisy = features[NOISE] > 0.5
            base = 0.6 if noisy else 0.02
            alt = (0.05 if noisy else 0.7) if helpful else base + 0.1
            jitter = lambda v: float(max(v + rng.normal(0, 0.005), 0.0))  # noqa: E731
            image = f"t{t}_{r}.png"
            text = {name: "x" for name in NAMES}
            if empty_alt and not noisy:
                text["p_alt"] = ""
            rows.append(
                {
                    "image": image,
                    "source": "synthetic",
                    "split": split,
                    "text_id": f"t{t}",
                    "lang": "en",
                    "content": "prose",
                    "font": "F",
                    "font_size": 12,
                    "theme": "light",
                    "degradation": "noise" if noisy else "none",
                    "scale": 1.0,
                    "pixels": 1000,
                    "gt": "x",
                    "features": [float(v) for v in features],
                    "cer": {
                        "p_base": jitter(base),
                        "p_alt": jitter(alt),
                        "p_same": jitter(base) + 0.01,
                        "p_bad": 0.9,
                    },
                    "text": text,
                    "conf_stats": {name: [0.9, 0.8, 0.8, 0.1, 0.1] for name in NAMES},
                }  # fmt: skip
            )
            if split != "train" or r == 0:
                timing[image] = {"features": 0.002,
                                 "time": {"p_base": 0.10, "p_alt": 0.12, "p_same": 0.10, "p_bad": 0.10}}  # fmt: skip
            if split != "browser":
                legacy[image] = {"split": split, "cer": [base, 0.3, 0.3], "action": 0,
                                 "time": [0.1, 0.05]}  # fmt: skip
    paths = {
        key: tmp_path / f"{key}.json" for key in ("results", "timing", "legacy", "model", "eval")
    }
    paths["results"].write_text(json.dumps({
        "feature_names": list(FEATURE_NAMES), "conf_stat_names": list(CONF_STAT_NAMES),
        "language": "eng",
        "pipelines": [{"name": name, "steps": ["light"], "psm": "6"} for name in NAMES],
        "rows": rows}))  # fmt: skip
    paths["timing"].write_text(json.dumps(
        {"loadavg_start": [0.1, 0.1, 0.1], "loadavg_end": [0.2, 0.1, 0.1], "rows": timing}))  # fmt: skip
    paths["legacy"].write_text(json.dumps(
        {"source": "t", "actions": ["tesseract", "easyocr", "merge"], "rows": legacy}))  # fmt: skip
    return paths


def arrays(paths):
    data = tr.load_arrays(paths["results"], paths["timing"])
    dev = np.isin(data["split"], tr.DEV_SPLITS)
    return data, dev, np.nanmean(data["seconds"][dev], axis=0)


def test_load_arrays(tmp_path):
    data, dev, mean_seconds = arrays(fake_run(tmp_path, empty_alt=True))
    n = len(data["rows"])
    assert data["names"] == list(NAMES)
    assert data["features"].shape == (n, len(FEATURE_NAMES)) and data["cer"].shape == (n, 4)
    assert data["conf_stats"].shape == (n, 4, 5)
    assert data["empty"][:, 1].any() and not data["empty"][:, 0].any()
    assert np.isnan(data["seconds"]).any() and not np.isnan(data["seconds"][~dev]).any()
    assert mean_seconds.tolist() == pytest.approx([0.10, 0.12, 0.10, 0.10])
    assert data["loadavg"] == [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]


def test_action_costs():
    assert tr.action_costs("pre_ocr", [0.1, 0.2, 0.3]).tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert tr.action_costs("cascade", [0.1, 0.2, 0.3]).tolist() == pytest.approx([0.1, 0.3, 0.4])


def test_design_before_ocr(tmp_path):
    data, dev, mean_seconds = arrays(fake_run(tmp_path, empty_alt=True))
    X, cer, seconds, names = tr.design(data, "pre_ocr", [0, 1], mean_seconds)
    assert X.shape[1] == len(FEATURE_NAMES) and names == list(FEATURE_NAMES)
    empty = data["empty"][:, 1]
    # an empty result is replaced by the default's text and both passes are paid
    assert np.allclose(cer[empty, 1], data["cer"][empty, 0])
    assert np.allclose(seconds[empty, 1], 0.22) and np.allclose(seconds[~empty, 1], 0.12)


def test_design_cascade_adds_the_default_confidences(tmp_path):
    data, dev, mean_seconds = arrays(fake_run(tmp_path))
    X, cer, seconds, names = tr.design(data, "cascade", [1, 0], mean_seconds)
    assert X.shape[1] == len(FEATURE_NAMES) + 5 and names[-5:] == list(CONF_STAT_NAMES)
    assert np.allclose(seconds[:, 0], 0.12) and np.allclose(seconds[:, 1], 0.22)
    assert np.allclose(cer[:, 0], data["cer"][:, 1])


def test_oof_predictions_never_fit_on_the_predicted_group(monkeypatch):
    seen = []

    def fake_fit(spec, X, y):
        seen.append(set(X[:, 0].astype(int)))
        return []

    monkeypatch.setattr(tr, "fit", fake_fit)
    monkeypatch.setattr(tr, "predict", lambda models, X: np.zeros((len(X), 2)))
    groups = np.repeat(np.arange(10), 3)
    X = np.column_stack([groups, np.zeros(30)])
    tr.oof_predictions({}, X, np.zeros((30, 2)), groups)
    assert len(seen) == 5 and all(len(fold) == 8 for fold in seen)


def test_select_actions_adds_the_complementary_pipeline_and_stops(tmp_path):
    data, dev, mean_seconds = arrays(fake_run(tmp_path))
    chosen, steps = tr.select_actions(data, dev, mean_seconds, spec=FAST[0])
    assert [data["names"][k] for k in chosen] == ["p_base", "p_alt"]
    assert steps[0]["added"] == "p_base" and steps[1]["added"] == "p_alt"
    assert steps[1]["oof_cer"] < steps[0]["oof_cer"] - 0.1
    assert steps[-1]["added"] is None and steps[-1]["rejected"] in ("p_same", "p_bad")


def test_select_actions_respects_the_cap(tmp_path):
    data, dev, mean_seconds = arrays(fake_run(tmp_path))
    chosen, steps = tr.select_actions(data, dev, mean_seconds, spec=FAST[0], max_actions=1)
    assert len(chosen) == 1 and len(steps) == 1


def test_select_actions_keeps_one_action_when_nothing_helps(tmp_path):
    data, dev, mean_seconds = arrays(fake_run(tmp_path, helpful=False))
    chosen, _ = tr.select_actions(data, dev, mean_seconds, spec=FAST[0])
    assert [data["names"][k] for k in chosen] == ["p_base"]


def test_pick_time_weight_takes_the_largest_weight_within_tolerance():
    y = np.array([[0.5, 0.1], [0.5, 0.1]])
    pred = y.copy()
    costs = np.array([0.1, 0.2])
    # weights 5 and 10 switch to the cheap action, which costs 0.4 CER
    assert tr.pick_time_weight(pred, y, costs, weights=[0.0, 5.0, 10.0]) == 0.0
    assert tr.pick_time_weight(pred, y, costs, weights=[0.0, 1.0, 10.0]) == 1.0
    assert tr.pick_time_weight(pred, y, costs, weights=[0.0, 1.0, 10.0], tolerance=0.5) == 10.0


def test_select_policy():
    close = {"pre_ocr": {"cer": 0.050, "time": 0.2}, "cascade": {"cer": 0.048, "time": 0.3}}
    assert tr.select_policy(close) == "pre_ocr"
    apart = {"pre_ocr": {"cer": 0.060, "time": 0.2}, "cascade": {"cer": 0.048, "time": 0.3}}
    assert tr.select_policy(apart) == "cascade"


def test_select_model_rejects_candidates_that_all_fail():
    with pytest.raises(ValueError):
        tr.select_model(np.zeros((4, 1)), np.zeros((4, 2)), np.arange(4), np.zeros(2), [])


def test_run_end_to_end(tmp_path):
    paths = fake_run(tmp_path)
    ev = tr.run(paths["results"], paths["timing"], paths["legacy"], paths["model"],
                paths["eval"], candidates=FAST)  # fmt: skip
    assert ev["shipped"] in ("pre_ocr", "cascade") and ev["best_static"] == "p_base"
    assert ev["actions"] == ["p_base", "p_alt"]
    router = Router(paths["model"])
    assert router.available and router.default.name == "p_base"
    assert set(ev["slices"]) == {"test", "unseen_font", "browser"}
    test = {s["policy"]: s for s in ev["slices"]["test"]}
    shipped = test[f"router_{ev['shipped']}"]
    assert shipped["cer"][0] < 0.1 < test["always_p_base"]["cer"][0]
    assert shipped["delta"][2] < 0 and ev["criteria"]["router_beats_best_static"]
    assert "router_v04" in test and "router_v04" not in {
        s["policy"] for s in ev["slices"]["browser"]
    }
    assert {"oracle", "oracle_pool"} <= set(test)
    assert set(ev["clean_vs_degraded"]) == {"clean", "degraded"}
    assert ev["easyocr"]["with_easyocr"] <= ev["easyocr"]["shipped"]
    assert json.loads(paths["eval"].read_text())["shipped"] == ev["shipped"]


def test_run_scores_the_shipped_policy_through_the_router(tmp_path, monkeypatch):
    calls = []
    original = Router.choose_vector

    def counting(self, x):
        calls.append(1)
        return original(self, x)

    monkeypatch.setattr(Router, "choose_vector", counting)
    paths = fake_run(tmp_path)
    tr.run(paths["results"], paths["timing"], paths["legacy"], paths["model"], paths["eval"],
           candidates=FAST)  # fmt: skip
    data = tr.load_arrays(paths["results"], paths["timing"])
    assert len(calls) >= int(np.isin(data["split"], tr.EVAL_SLICES).sum())


def test_run_ships_a_static_model_when_routing_does_not_help(tmp_path):
    paths = fake_run(tmp_path, helpful=False)
    ev = tr.run(paths["results"], paths["timing"], paths["legacy"], paths["model"],
                paths["eval"], candidates=FAST)  # fmt: skip
    model = load_model(paths["model"])
    assert ev["shipped"] == "static" and ev["actions"] == ["p_base"]
    assert model.policy == "static" and len(model.actions) == 1 and not model.regressors
    assert not ev["criteria"]["router_beats_best_static"]
    assert [s["policy"] for s in ev["slices"]["test"]][0] == "always_p_base"


def test_the_model_does_not_depend_on_evaluation_slices(tmp_path):
    first = fake_run(tmp_path / "a")
    tr.run(first["results"], first["timing"], first["legacy"], first["model"], first["eval"],
           candidates=FAST)  # fmt: skip
    second = fake_run(tmp_path / "b")
    payload = json.loads(second["results"].read_text())
    for row in payload["rows"]:
        if row["split"] not in tr.DEV_SPLITS:
            row["cer"] = {name: 0.5 for name in NAMES}
    second["results"].write_text(json.dumps(payload))
    tr.run(second["results"], second["timing"], second["legacy"], second["model"],
           second["eval"], candidates=FAST)  # fmt: skip
    assert first["model"].read_bytes() == second["model"].read_bytes()


def test_delivered_applies_the_empty_text_rule(tmp_path):
    paths = fake_run(tmp_path, empty_alt=True)
    tr.run(paths["results"], paths["timing"], paths["legacy"], paths["model"], paths["eval"],
           candidates=FAST)  # fmt: skip
    data = tr.load_arrays(paths["results"], paths["timing"])
    mask = data["split"] == "val"
    columns = tr.delivered(data, Router(paths["model"]), mask)
    assert set(columns.tolist()) <= {0, 1}
    assert not data["empty"][mask][np.arange(mask.sum()), columns].any()


def test_images_above_the_size_limit_are_scored_with_the_default(tmp_path):
    paths = fake_run(tmp_path)
    payload = json.loads(paths["results"].read_text())
    for row in payload["rows"]:
        if row["split"] == "test":
            row["pixels"] = MAX_ROUTED_PIXELS + 1
    paths["results"].write_text(json.dumps(payload))
    ev = tr.run(paths["results"], paths["timing"], paths["legacy"], paths["model"], paths["eval"],
                candidates=FAST)  # fmt: skip
    test = {s["policy"]: s for s in ev["slices"]["test"]}
    for policy in ("router_pre_ocr", "router_cascade"):
        assert test[policy]["share"] == {"p_base": 1.0, "p_alt": 0.0}
        assert test[policy]["cer"][0] == pytest.approx(test["always_p_base"]["cer"][0])
        assert test[policy]["time"] == pytest.approx(test["always_p_base"]["time"])
    other = {s["policy"]: s for s in ev["slices"]["unseen_font"]}
    assert other[f"router_{ev['shipped']}"]["share"]["p_alt"] > 0
    data = tr.load_arrays(paths["results"], paths["timing"])
    mask = data["split"] == "test"
    assert ev["dataset"]["unrouted"] == int(mask.sum())
    assert set(tr.delivered(data, Router(paths["model"]), mask).tolist()) == {0}


def test_images_above_the_size_limit_are_scored_with_the_v04_pipeline_when_pooled(tmp_path):
    paths = fake_run(tmp_path)
    for key in ("results", "timing"):
        paths[key].write_text(paths[key].read_text().replace("p_bad", V04.name))
    payload = json.loads(paths["results"].read_text())
    for row in payload["rows"]:
        if row["split"] == "test":
            row["pixels"] = MAX_ROUTED_PIXELS + 1
    paths["results"].write_text(json.dumps(payload))
    ev = tr.run(paths["results"], paths["timing"], paths["legacy"], paths["model"], paths["eval"],
                candidates=FAST)  # fmt: skip
    shipped = next(s for s in ev["slices"]["test"] if s["policy"] == f"router_{ev['shipped']}")
    assert shipped["share"] == {"p_base": 0.0, "p_alt": 0.0, V04.name: 1.0}
    assert shipped["cer"][0] == pytest.approx(0.9)
    data = tr.load_arrays(paths["results"], paths["timing"])
    columns = tr.delivered(data, Router(paths["model"]), data["split"] == "test")
    assert set(columns.tolist()) == {data["names"].index(V04.name)}
