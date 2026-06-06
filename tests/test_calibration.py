"""Calibration metric + isotonic-refit behaviour, independent of OCR runs."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from benchmarks.calib_merge import _calibrate
from benchmarks.calibration import _domain, _ece
from benchmarks.run_eval import _label_correct


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
    rows = [{"conf": float(c), "correct": int(y)} for c, y in zip(conf, correct)]
    res = _domain(rows)
    assert res["ece_cal"] < res["ece_raw"]


def test_calibrate_preserves_line_shape_and_range():
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        np.array([0.0, 0.5, 1.0]), np.array([0, 0, 1])
    )
    cal = _calibrate([[0.2, 0.9], [0.5], []], iso)
    assert [len(line) for line in cal] == [2, 1, 0]
    assert all(0.0 <= v <= 1.0 for line in cal for v in line)
