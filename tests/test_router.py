"""Tests for the engine router (no OCR)."""

import sys

import numpy as np
import pytest

from sniptext import router
from sniptext.router import (
    ACTIONS,
    Router,
    RouterModel,
    action_costs,
    choose_actions,
    conf_stats,
    read_table,
    write_table,
)

RIDGE = {"kind": "ridge", "alpha": 0.1}


def make_table(path, policy="pre_ocr", time_weight=0.0, n=80):
    """Tesseract is good when x0 is small, EasyOCR when x0 is large, merge is always mediocre."""
    rng = np.random.default_rng(0)
    x0 = rng.random(n)
    names = ["x0", "x1"]
    cols = [x0, rng.random(n)]
    if policy == "cascade":
        names += list(router.CONF_STAT_NAMES)
        cols += [1.0 - x0, 1.0 - x0, 1.0 - x0, x0, np.full(n, 0.2)]
    X = np.column_stack(cols)
    y = np.column_stack([x0, 1.0 - x0, np.full(n, 0.45)])
    meta = {
        "policy": policy,
        "model": RIDGE,
        "time": {"tesseract": 0.1, "easyocr": 0.4},
        "time_weight": time_weight,
        "tesseract_call": "detailed",
    }
    write_table(path, names, X, y, meta)
    return path


class TestConfStats:
    def test_values(self):
        stats = conf_stats([[0.9, 0.5], [0.7]])
        assert stats.shape == (5,)
        assert stats[0] == pytest.approx(0.7)
        assert stats[1] == pytest.approx(0.5)
        assert stats[3] == pytest.approx(1 / 3)
        assert stats[4] == pytest.approx(0.03)

    def test_conf_stats_empty(self):
        for empty in (None, [], [[]]):
            assert conf_stats(empty).tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


class TestCostsAndChoice:
    def test_action_costs(self):
        assert action_costs("pre_ocr", 0.1, 0.4).tolist() == pytest.approx([0.1, 0.4, 0.5])
        assert action_costs("cascade", 0.1, 0.4).tolist() == pytest.approx([0.1, 0.5, 0.5])
        with pytest.raises(ValueError):
            action_costs("other", 0.1, 0.4)

    def test_picks_lowest_predicted_cer(self):
        pred = np.array([[0.1, 0.5, 0.3], [0.6, 0.2, 0.3]])
        assert choose_actions(pred, np.array([0.1, 0.4, 0.5]), 0.0).tolist() == [0, 1]

    def test_time_weight_shifts_choice_to_cheaper_action(self):
        pred = np.array([[0.30, 0.25, 0.26]])
        costs = np.array([0.1, 0.4, 0.5])
        assert choose_actions(pred, costs, 0.0).tolist() == [1]
        assert choose_actions(pred, costs, 1.0).tolist() == [0]

    def test_tie_goes_to_cheaper_then_earlier(self):
        pred = np.array([[0.2, 0.2, 0.2]])
        assert choose_actions(pred, np.array([0.3, 0.1, 0.5]), 0.0).tolist() == [1]
        assert choose_actions(pred, np.array([0.1, 0.5, 0.5]), 0.0).tolist() == [0]
        assert choose_actions(
            np.array([[0.9, 0.2, 0.2]]), np.array([0.1, 0.5, 0.5]), 0.0
        ).tolist() == [1]


class TestTable:
    def test_round_trip(self, tmp_path):
        path = make_table(tmp_path / "t.csv.gz")
        table = read_table(path)
        assert table.feature_names == ("x0", "x1")
        assert table.X.shape == (80, 2)
        assert table.y.shape == (80, 3)
        assert table.meta["policy"] == "pre_ocr"
        assert len(table.digest) == 64

    def test_digest_is_stable_and_content_sensitive(self, tmp_path):
        a = read_table(make_table(tmp_path / "a.csv.gz")).digest
        b = read_table(make_table(tmp_path / "b.csv.gz")).digest
        c = read_table(make_table(tmp_path / "c.csv.gz", n=81)).digest
        assert a == b
        assert a != c

    def test_rejects_other_files(self, tmp_path):
        import gzip

        bad = tmp_path / "bad.csv.gz"
        bad.write_bytes(gzip.compress(b"a,b\n1,2\n"))
        with pytest.raises(ValueError):
            read_table(bad)
        with pytest.raises(OSError):
            read_table(tmp_path / "missing.csv.gz")


class TestRouterModel:
    def test_predictions_are_clipped(self):
        X = np.linspace(0, 1, 50)[:, None]
        y = np.column_stack([X[:, 0] * 3 - 1, 1 - X[:, 0], np.full(50, 0.5)])
        pred = RouterModel(RIDGE).fit(X, y).predict_cer(X)
        assert pred.shape == (50, 3)
        assert pred.min() >= 0.0 and pred.max() <= 1.0

    def test_gbr_spec(self):
        X = np.random.default_rng(0).random((60, 2))
        y = np.column_stack([X[:, 0], 1 - X[:, 0], np.full(60, 0.5)])
        spec = {"kind": "gbr", "n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}
        assert RouterModel(spec).fit(X, y).predict_cer(X).shape == (60, 3)

    def test_unknown_spec(self):
        with pytest.raises(ValueError):
            RouterModel({"kind": "other"}).fit(np.zeros((4, 1)), np.zeros((4, 3)))


class TestRouter:
    def test_chooses_by_features(self, tmp_path):
        r = Router(make_table(tmp_path / "t.csv.gz"), cache_dir=tmp_path / "cache")
        assert r.available
        assert r.policy == "pre_ocr"
        assert r.tesseract_call == "detailed"
        assert r.choose(np.array([0.05, 0.5])) == "tesseract"
        assert r.choose(np.array([0.95, 0.5])) == "easyocr"

    def test_time_weight_override(self, tmp_path):
        path = make_table(tmp_path / "t.csv.gz")
        x = np.array([0.6, 0.5])
        assert Router(path, cache_dir=tmp_path / "c").choose(x) == "easyocr"
        assert Router(path, cache_dir=tmp_path / "c", time_weight=5.0).choose(x) == "tesseract"

    def test_cascade_uses_tesseract_confidence(self, tmp_path):
        r = Router(make_table(tmp_path / "t.csv.gz", policy="cascade"), cache_dir=tmp_path / "c")
        assert r.policy == "cascade"
        x = np.array([0.5, 0.5])
        assert r.choose(x, [[0.95, 0.97]]) == "tesseract"
        assert r.choose(x, [[0.05, 0.10]]) == "easyocr"
        assert r.choose(x, None) in ACTIONS

    def test_cache_is_written_and_reused(self, tmp_path):
        path = make_table(tmp_path / "t.csv.gz")
        cache = tmp_path / "cache"
        Router(path, cache_dir=cache).choose(np.array([0.1, 0.5]))
        files = list(cache.glob("router-*.pkl"))
        assert len(files) == 1
        stamp = files[0].stat().st_mtime_ns
        Router(path, cache_dir=cache).choose(np.array([0.1, 0.5]))
        assert files[0].stat().st_mtime_ns == stamp

    def test_new_table_gets_new_cache_entry(self, tmp_path):
        cache = tmp_path / "cache"
        Router(make_table(tmp_path / "a.csv.gz"), cache_dir=cache).choose(np.array([0.1, 0.5]))
        Router(make_table(tmp_path / "b.csv.gz", n=81), cache_dir=cache).choose(
            np.array([0.1, 0.5])
        )
        assert len(list(cache.glob("router-*.pkl"))) == 2

    def test_corrupt_cache_is_refitted(self, tmp_path):
        path = make_table(tmp_path / "t.csv.gz")
        cache = tmp_path / "cache"
        Router(path, cache_dir=cache).choose(np.array([0.1, 0.5]))
        next(cache.glob("router-*.pkl")).write_bytes(b"not a pickle")
        assert Router(path, cache_dir=cache).choose(np.array([0.05, 0.5])) == "tesseract"

    def test_unwritable_cache_dir_still_works(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        r = Router(make_table(tmp_path / "t.csv.gz"), cache_dir=blocker / "sub")
        assert r.choose(np.array([0.95, 0.5])) == "easyocr"

    def test_missing_table_falls_back(self, tmp_path):
        r = Router(tmp_path / "missing.csv.gz", cache_dir=tmp_path / "c")
        assert not r.available
        assert r.choose(np.array([0.9, 0.5])) == "tesseract"

    def test_feature_count_mismatch_falls_back(self, tmp_path):
        r = Router(make_table(tmp_path / "t.csv.gz"), cache_dir=tmp_path / "c")
        assert r.choose(np.array([0.9, 0.5, 0.1])) == "tesseract"

    def test_without_sklearn_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "sklearn", None)
        r = Router(make_table(tmp_path / "t.csv.gz"), cache_dir=tmp_path / "c")
        assert not r.available
        assert r.choose(np.array([0.9, 0.5])) == "tesseract"


class TestShippedTable:
    def test_shipped_table_matches_the_analyzer(self, tmp_path):
        from PIL import Image

        from sniptext.analyzer import FEATURE_NAMES, ImageAnalyzer

        table = read_table(router._TABLE_PATH)
        expected = list(FEATURE_NAMES)
        if table.meta["policy"] == "cascade":
            expected += list(router.CONF_STAT_NAMES)
        assert list(table.feature_names) == expected
        assert table.meta["tesseract_call"] in ("detailed", "plain")
        assert table.meta["time_weight"] >= 0.0
        assert len(table.X) > 1000

        r = Router(cache_dir=tmp_path)
        assert r.available
        features = ImageAnalyzer().extract_features(Image.new("RGB", (400, 120), (250, 250, 250)))
        assert r.choose(features, [[0.9, 0.8]]) in ACTIONS
