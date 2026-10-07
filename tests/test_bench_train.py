"""Router selection and export on a small fabricated results file (no OCR)."""

import json

import numpy as np
import pytest

from benchmarks import train_router
from sniptext.analyzer import FEATURE_NAMES
from sniptext.router import ACTIONS, CONF_STAT_NAMES, Router

RIDGE_ONLY = [{"kind": "ridge", "alpha": 1.0}]


def fake_results(path, n_texts=60, seed=0):
    """Tesseract is good when feature 0 is small, EasyOCR when it is large."""
    rng = np.random.default_rng(seed)
    rows = []

    def row(image, split, text_id):
        features = rng.random(len(FEATURE_NAMES))
        tess = float(np.clip(features[0] + rng.normal(0, 0.05), 0, 1.5))
        easy = float(np.clip(1 - features[0] + rng.normal(0, 0.05), 0, 1.5))
        return {
            "image": image,
            "source": "sroie" if split == "ood" else "synthetic",
            "split": split,
            "text_id": text_id,
            "lang": "en" if rng.random() < 0.5 else "ru",
            "content": "prose",
            "font": "f",
            "font_size": 16,
            "theme": "light" if rng.random() < 0.5 else "dark",
            "degradation": str(rng.choice(["none", "blur", "blur+noise"])),
            "features": features.tolist(),
            "legacy_features": rng.random(7).tolist(),
            "conf_stats": [1 - tess / 1.5, 0.5, 0.5, tess / 1.5, 0.1],
            "cer": {"tesseract_plain": tess, "tesseract": tess, "easyocr": easy, "merge": 0.4},
            "wer": {"tesseract_plain": tess, "tesseract": tess, "easyocr": easy, "merge": 0.4},
            "time": {"tesseract_plain": 0.1, "tesseract": 0.12, "easyocr": 0.3, "features": 0.002},
            "weak": {"fast_quality": float(rng.random()), "ens_quality": float(rng.random())},
        }

    for t in range(n_texts):
        split = "train" if t < n_texts * 0.6 else "val" if t < n_texts * 0.8 else "test"
        for k in range(4):
            rows.append(row(f"t{t}_{k}.png", split, f"text{t}"))
        if split == "test":
            rows.append(row(f"t{t}_u.png", "unseen_font", f"text{t}"))
    for k in range(10):
        rows.append(row(f"r{k}.jpg", "ood", f"r{k}.jpg"))
    path.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "conf_stat_names": list(CONF_STAT_NAMES),
                "language": "eng+rus",
                "rows": rows,
            }
        )
    )
    return path


def test_tesseract_call_rule(tmp_path):
    data = train_router.load_arrays(fake_results(tmp_path / "r.json"))
    dev = np.isin(data["split"], ("train", "val"))
    assert train_router.tesseract_call_rule(data, dev) == "detailed"
    data["cer"]["tesseract"] = data["cer"]["tesseract"] + 0.05
    assert train_router.tesseract_call_rule(data, dev) == "plain"


def test_design_shapes_and_columns(tmp_path):
    data = train_router.load_arrays(fake_results(tmp_path / "r.json"))
    n = len(data["rows"])
    X, cer, times, names = train_router.design(data, "pre_ocr", "detailed")
    assert X.shape == (n, len(FEATURE_NAMES)) and cer.shape == (n, 3) and times.shape == (n, 3)
    assert names == list(FEATURE_NAMES)
    assert times[0] == pytest.approx([0.12, 0.3, 0.42])
    X, cer, times, names = train_router.design(data, "cascade", "plain")
    assert X.shape == (n, len(FEATURE_NAMES) + len(CONF_STAT_NAMES))
    assert names == list(FEATURE_NAMES) + list(CONF_STAT_NAMES)
    assert times[0] == pytest.approx([0.12, 0.42, 0.42])
    _, _, times, _ = train_router.design(data, "pre_ocr", "plain")
    assert times[0] == pytest.approx([0.1, 0.3, 0.4])


def test_oof_predictions_never_fit_on_the_predicted_group(monkeypatch):
    seen = []

    class Spy:
        def __init__(self, spec):
            self.groups = set()

        def fit(self, X, y):
            self.groups = set(X[:, 0].tolist())
            return self

        def predict_cer(self, X):
            seen.append((self.groups, set(X[:, 0].tolist())))
            return np.full((len(X), 3), 0.5)

    monkeypatch.setattr(train_router, "RouterModel", Spy)
    groups = np.repeat(np.arange(10), 4)
    pred = train_router.oof_predictions(
        {}, groups[:, None].astype(float), np.zeros((40, 3)), groups
    )
    assert pred.shape == (40, 3) and np.all(pred == 0.5)
    assert len(seen) == 5
    for fitted_on, predicted in seen:
        assert not fitted_on & predicted


def test_pick_time_weight_takes_the_largest_weight_within_tolerance():
    costs = np.array([0.1, 0.4, 0.5])
    y = np.array([[0.30, 0.25, 0.90]])
    # Switching to Tesseract costs 0.05 CER, more than the 0.005 tolerance; it
    # happens once 0.30 + 0.1 w < 0.25 + 0.4 w, that is w > 1/6.
    assert train_router.pick_time_weight(y, y, costs, weights=[0.0, 0.1, 0.2, 1.0]) == 0.1
    y = np.array([[0.252, 0.250, 0.90]])
    assert train_router.pick_time_weight(y, y, costs, weights=[0.0, 0.1, 0.2, 1.0]) == 1.0


def test_select_policy():
    pick = train_router.select_policy
    assert (
        pick({"pre_ocr": {"cer": 0.20, "time": 0.2}, "cascade": {"cer": 0.10, "time": 0.4}})
        == "cascade"
    )
    assert (
        pick({"pre_ocr": {"cer": 0.10, "time": 0.2}, "cascade": {"cer": 0.20, "time": 0.1}})
        == "pre_ocr"
    )
    assert (
        pick({"pre_ocr": {"cer": 0.103, "time": 0.2}, "cascade": {"cer": 0.100, "time": 0.4}})
        == "pre_ocr"
    )
    assert (
        pick({"pre_ocr": {"cer": 0.100, "time": 0.5}, "cascade": {"cer": 0.103, "time": 0.4}})
        == "cascade"
    )


def test_run_end_to_end(tmp_path):
    results = fake_results(tmp_path / "r.json")
    table = tmp_path / "pkg" / "router_train.csv.gz"
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"n": 5, "easyocr_cpu_mean": 3.0, "easyocr_gpu_mean": 0.3}))
    ev = train_router.run(results, table, tmp_path / "eval.json", timing, candidates=RIDGE_ONLY)

    assert json.loads((tmp_path / "eval.json").read_text())["shipped"] == ev["shipped"]
    assert ev["shipped"] in train_router.POLICIES
    assert ev["best_static"] in ACTIONS
    assert ev["dataset"]["by_split"] == {
        "train": 144,
        "val": 48,
        "test": 48,
        "unseen_font": 12,
        "ood": 10,
    }
    assert ev["dataset"]["texts"] == 60

    names = [s["policy"] for s in ev["slices"]["test"]]
    assert names == [
        "always_tesseract",
        "always_easyocr",
        "always_merge",
        "legacy_rules",
        "router_pre_ocr",
        "router_cascade",
        "oracle",
    ]
    by_name = {s["policy"]: s for s in ev["slices"]["test"]}
    assert by_name["oracle"]["regret"][0] == 0.0
    # The signal is strong by construction: both routers beat every static policy.
    for policy in train_router.POLICIES:
        assert by_name[f"router_{policy}"]["cer"][0] < by_name["always_merge"]["cer"][0]
        assert len(ev["policies"][policy]["selection"]) == 1
        assert set(ev["policies"][policy]["ablation"]) == set(
            ev["policies"][policy]["feature_names"]
        )
        assert len(ev["policies"][policy]["curve"]) == len(train_router.TIME_WEIGHTS)
    assert set(ev["slices"]) == {"test", "unseen_font", "ood"}
    assert set(ev["breakdown"]) == {"degradation", "theme", "lang", "content"}
    assert set(ev["breakdown"]["degradation"]) == {"none", "blur", "two combined"}
    assert ev["cpu"]["ratio"] == pytest.approx(10.0)
    assert set(ev["weak_labels"]) == {"synthetic", "all"}

    router = Router(table, cache_dir=tmp_path / "cache")
    assert router.available
    assert router.policy == ev["shipped"]
    expected = len(FEATURE_NAMES) + (len(CONF_STAT_NAMES) if ev["shipped"] == "cascade" else 0)
    assert router.choose(np.full(len(FEATURE_NAMES), 0.5), [[0.9]]) in ACTIONS
    assert len(ev["policies"][ev["shipped"]]["feature_names"]) == expected


def test_run_without_cpu_timing(tmp_path):
    ev = train_router.run(
        fake_results(tmp_path / "r.json"),
        tmp_path / "t.csv.gz",
        tmp_path / "eval.json",
        tmp_path / "missing.json",
        candidates=RIDGE_ONLY,
    )
    assert ev["cpu"] is None
