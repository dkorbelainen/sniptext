"""Calibration metric + isotonic-refit behaviour, independent of OCR runs."""

import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

from benchmarks import calib_merge, calibration
from benchmarks.calib_merge import _calibrate, _label_correct
from benchmarks.calibration import _domain, _ece, reliability


def test_ece_perfectly_calibrated_is_zero():
    # Confidence 0.7 with exactly 70% accuracy in that bin -> no gap.
    conf = np.full(100, 0.75)
    correct = np.array([1] * 75 + [0] * 25)
    assert _ece(conf, correct) < 1e-9


def test_ece_overconfident_is_large():
    conf = np.full(100, 0.95)
    correct = np.array([1] * 50 + [0] * 50)
    assert _ece(conf, correct) > 0.4


def test_ece_in_unit_range():
    rng = np.random.default_rng(0)
    conf = rng.random(500)
    correct = (rng.random(500) < 0.5).astype(int)
    assert 0.0 <= _ece(conf, correct) <= 1.0


def test_label_correct_aligns_to_gt():
    assert _label_correct(["the", "cat", "sat"], ["the", "dog", "sat"]) == [1, 0, 1]
    assert _label_correct(["a", "b"], ["a", "b"]) == [1, 1]
    assert _label_correct(["x"], ["y"]) == [0]


def test_isotonic_refit_reduces_ece_on_overconfident_domain():
    # Over-confident domain: high stated confidence, ~50% true accuracy.
    rng = np.random.default_rng(42)
    n = 2000
    conf = rng.uniform(0.8, 1.0, n)
    correct = (rng.random(n) < 0.5).astype(int)
    rows = [
        {"conf": float(c), "correct": int(y), "image": f"img{i // 20}"}
        for i, (c, y) in enumerate(zip(conf, correct))
    ]
    res = _domain(rows)
    assert res["ece_cal"] < res["ece_raw"]


def test_calibrate_preserves_line_shape_and_range():
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        np.array([0.0, 0.5, 1.0]), np.array([0, 0, 1])
    )
    cal = _calibrate([[0.2, 0.9], [0.5], []], iso)
    assert [len(line) for line in cal] == [2, 1, 0]
    assert all(0.0 <= v <= 1.0 for line in cal for v in line)


def test_domain_split_keeps_an_image_on_one_side(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def fit(self, conf, y):
            seen["fit"] = set(conf.tolist())
            return self

        def predict(self, conf):
            seen["predict"] = set(conf.tolist())
            return conf

    monkeypatch.setattr(calibration, "IsotonicRegression", Recorder)
    # Every word of image k carries confidence k / 100, so a value identifies its image.
    rows = [
        {"conf": k / 100, "correct": (k + j) % 2, "image": f"img{k}"}
        for k in range(40)
        for j in range(5)
    ]
    calibration._domain(rows)
    assert seen["fit"] and seen["predict"]
    assert not seen["fit"] & seen["predict"]


def test_domain_with_a_single_image_reports_raw_only():
    rows = [{"conf": 0.9, "correct": i % 2, "image": "only"} for i in range(10)]
    res = _domain(rows)
    assert res["n"] == 10
    assert res["ece_cal"] != res["ece_cal"]  # NaN: nothing to hold out


def test_reliability_bins():
    rows = [{"conf": 0.95, "correct": 1}] * 8 + [{"conf": 0.95, "correct": 0}] * 2
    rows += [{"conf": 0.15, "correct": 0}] * 5
    bins = reliability(rows)
    assert len(bins) == 2
    assert bins[0] == {"conf": 0.15, "accuracy": 0.0, "n": 5}
    assert bins[1]["n"] == 10 and abs(bins[1]["accuracy"] - 0.8) < 1e-9


def test_calib_merge_split_keeps_a_text_on_one_side(tmp_path, monkeypatch):
    rows = [
        {
            "source": "synthetic",
            "text_id": f"text{t}",
            "image": f"text{t}_{k}.png",
            "gt": "hello world",
            "tess_text": "hello wor1d",
            "tess_conf": [[0.9, 0.3]],
            "easy_text": "hello world",
            "easy_conf": [[0.8, 0.8]],
        }
        for t in range(20)
        for k in range(3)
    ]
    path = tmp_path / "merge_inputs.json"
    path.write_text(json.dumps(rows))
    monkeypatch.setattr(calib_merge, "_MERGE_INPUTS", path)
    seen = {}
    original_fit, original_domain = calib_merge._fit_calibrators, calib_merge._domain

    def fit(train):
        seen["train"] = {r["text_id"] for r in train}
        return original_fit(train)

    def domain(test, isos):
        seen["test"] = {r["text_id"] for r in test}
        return original_domain(test, isos)

    monkeypatch.setattr(calib_merge, "_fit_calibrators", fit)
    monkeypatch.setattr(calib_merge, "_domain", domain)
    out = calib_merge.evaluate()
    assert seen["train"] and seen["test"]
    assert not seen["train"] & seen["test"]
    assert out["synthetic"]["n_test"] == 18
