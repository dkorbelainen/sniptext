"""Benchmark plumbing that needs no OCR: engine runner, legacy baseline, row building."""

import numpy as np
import pytest
from PIL import Image

from benchmarks import engines
from benchmarks.engines import EngineRunner, RunResult
from benchmarks.evaluate import (
    cluster_bootstrap,
    realized,
    summarize,
    time_matrix,
    weak_label_agreement,
)
from benchmarks.legacy_policy import legacy_actions, legacy_features
from benchmarks.run_eval import build_row
from sniptext.analyzer import FEATURE_NAMES, ImageAnalyzer
from sniptext.metrics import OCRQualityMetrics
from sniptext.router import CONF_STAT_NAMES


class StubBackend:
    def __init__(self, plain, detailed, confs):
        self._plain, self._detailed, self._confs = plain, detailed, confs
        self.calls = []

    def recognize(self, image):
        self.calls.append("plain")
        return self._plain

    def recognize_detailed(self, image):
        self.calls.append("detailed")
        return self._detailed, self._confs


def stub_runner():
    runner = EngineRunner.__new__(EngineRunner)
    runner.tess = StubBackend("hello wor1d", "hello wor1d", [[0.9, 0.3]])
    runner.easy = StubBackend("unused", "hello world", [[0.8, 0.8]])
    runner.ensemble = engines.EnsembleOCR()
    return runner


def test_run_all_runs_each_engine_once_and_merges_by_confidence():
    runner = stub_runner()
    result = runner.run_all(Image.new("RGB", (40, 20)))
    assert runner.tess.calls == ["plain", "detailed"]
    assert runner.easy.calls == ["detailed"]
    assert set(result.texts) == {"tesseract_plain", "tesseract", "easyocr", "merge"}
    assert result.texts["merge"] == "hello world"
    assert set(result.times) == {"tesseract_plain", "tesseract", "easyocr"}
    assert all(t >= 0.0 for t in result.times.values())
    assert result.confs["tesseract"] == [[0.9, 0.3]]


def test_legacy_features_keep_the_old_seven():
    assert legacy_features(Image.new("RGB", (300, 100), (240, 240, 240))).shape == (7,)


def test_legacy_actions_follow_the_old_rules():
    # brightness, contrast, sharpness, has_color, size_ratio, text_density, noise_level
    clear = [0.8, 0.9, 0.9, 0.0, 0.5, 0.2, 0.05]
    noisy = [0.5, 0.4, 0.5, 0.0, 0.5, 0.2, 0.9]
    empty = [0.5, 0.6, 0.6, 0.0, 0.5, 0.0, 0.1]
    actions = legacy_actions(np.array([clear, noisy, empty]))
    assert actions.tolist() == [0, 2, 2]


def test_legacy_actions_are_deterministic_for_a_seed():
    borderline = np.random.default_rng(0).uniform(0.3, 0.5, size=(40, 7))
    assert (
        legacy_actions(borderline, seed=3).tolist() == legacy_actions(borderline, seed=3).tolist()
    )
    assert set(legacy_actions(borderline, seed=3).tolist()) <= {0, 2}


def sample(tmp_path):
    return {
        "path": tmp_path / "img.png",
        "gt": "hello world",
        "source": "synthetic",
        "split": "test",
        "text_id": "abc",
        "lang": "en",
        "content": "prose",
        "font": "DejaVu Sans",
        "font_size": 18,
        "theme": "dark",
        "degradation": "blur+noise",
    }


def test_build_row(tmp_path):
    image = Image.new("RGB", (200, 60), (30, 30, 30))
    result = RunResult(
        texts={
            "tesseract_plain": "hello wor1d",
            "tesseract": "hello wor1d",
            "easyocr": "hello world",
            "merge": "hello world",
        },
        confs={"tesseract": [[0.9, 0.3]], "easyocr": [[0.8, 0.8]]},
        times={"tesseract_plain": 0.1, "tesseract": 0.12, "easyocr": 0.3},
    )
    row, merge_input, word_conf = build_row(
        sample(tmp_path), image, result, ImageAnalyzer(), OCRQualityMetrics()
    )
    assert row["image"] == "img.png"
    assert row["split"] == "test" and row["text_id"] == "abc"
    assert len(row["features"]) == len(FEATURE_NAMES)
    assert len(row["legacy_features"]) == 7
    assert len(row["conf_stats"]) == len(CONF_STAT_NAMES)
    assert row["cer"]["easyocr"] == 0.0
    assert row["cer"]["tesseract"] == 1 / 11
    assert row["wer"]["tesseract"] == 0.5
    assert set(row["time"]) == {"tesseract_plain", "tesseract", "easyocr", "features"}
    assert 0.0 <= row["weak"]["fast_quality"] <= 1.0
    assert 0.0 <= row["weak"]["ens_quality"] <= 1.0
    assert merge_input["tess_conf"] == [[0.9, 0.3]]
    assert merge_input["text_id"] == "abc" and merge_input["image"] == "img.png"
    assert [w["correct"] for w in word_conf if w["engine"] == "tesseract"] == [1, 0]
    assert all(w["image"] == "img.png" for w in word_conf)


def test_build_row_with_no_recognised_text(tmp_path):
    image = Image.new("RGB", (200, 60), (30, 30, 30))
    result = RunResult(
        texts={"tesseract_plain": "", "tesseract": "", "easyocr": "", "merge": ""},
        confs={"tesseract": None, "easyocr": None},
        times={"tesseract_plain": 0.1, "tesseract": 0.1, "easyocr": 0.2},
    )
    row, _, word_conf = build_row(
        sample(tmp_path), image, result, ImageAnalyzer(), OCRQualityMetrics()
    )
    assert row["cer"] == {"tesseract_plain": 1.0, "tesseract": 1.0, "easyocr": 1.0, "merge": 1.0}
    assert row["conf_stats"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert word_conf == []


def test_bootstrap_of_a_constant_is_that_constant():
    assert cluster_bootstrap([0.3] * 12, list(range(12))) == pytest.approx((0.3, 0.3, 0.3))


def test_bootstrap_interval_brackets_the_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    values = rng.random(300)
    clusters = np.arange(300)
    mean, low, high = cluster_bootstrap(values, clusters)
    assert low < mean < high
    assert cluster_bootstrap(values, clusters) == (mean, low, high)
    assert cluster_bootstrap(values, clusters, seed=5) != (mean, low, high)


def test_clustered_values_give_a_wider_interval():
    rng = np.random.default_rng(2)
    per_cluster = rng.random(50)
    values = np.repeat(per_cluster, 6)
    _, low_c, high_c = cluster_bootstrap(values, np.repeat(np.arange(50), 6))
    _, low_i, high_i = cluster_bootstrap(values, np.arange(300))
    assert (high_c - low_c) > 1.5 * (high_i - low_i)


def test_realized_and_time_matrix():
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert realized(matrix, np.array([2, 0])).tolist() == [3.0, 4.0]
    t_tess, t_easy = np.array([0.1, 0.2]), np.array([0.4, 0.5])
    assert time_matrix("pre_ocr", t_tess, t_easy) == pytest.approx(
        np.array([[0.1, 0.4, 0.5], [0.2, 0.5, 0.7]])
    )
    assert time_matrix("cascade", t_tess, t_easy) == pytest.approx(
        np.array([[0.1, 0.5, 0.5], [0.2, 0.7, 0.7]])
    )


def test_summarize():
    cer = np.array([[0.0, 0.5, 0.2], [2.0, 0.1, 0.3], [0.4, 0.4, 0.0], [0.3, 0.9, 0.6]])
    times = time_matrix("pre_ocr", np.full(4, 0.1), np.full(4, 0.4))
    clusters = np.array(["a", "a", "b", "c"])
    oracle = summarize("oracle", cer, np.minimum(cer, 1.0).argmin(axis=1), times, clusters)
    assert oracle["regret"][0] == 0.0
    assert oracle["cer"][0] == pytest.approx(0.1)
    always = summarize("always_tesseract", cer, np.zeros(4, dtype=int), times, clusters,
                       reference=np.minimum(cer[:, 1], 1.0))  # fmt: skip
    assert always["n"] == 4
    assert always["cer"][0] == pytest.approx((0.0 + 1.0 + 0.4 + 0.3) / 4)
    assert always["cer_unclipped"] == pytest.approx((0.0 + 2.0 + 0.4 + 0.3) / 4)
    assert always["cer_median"] == pytest.approx(0.35)
    assert always["share"] == {"tesseract": 1.0, "easyocr": 0.0, "merge": 0.0}
    assert always["time"] == pytest.approx(0.1)
    assert always["delta"][0] == pytest.approx((1.7 - 1.9) / 4)
    assert "delta" not in oracle


def weak_row(fast, ens, cer_tess, cer_merge):
    return {
        "weak": {"fast_quality": fast, "ens_quality": ens},
        "cer": {"tesseract_plain": cer_tess, "merge": cer_merge},
    }


def test_weak_label_agreement_perfect_and_useless():
    perfect = [weak_row(0.9, 0.5, 0.0, 0.3), weak_row(0.4, 0.8, 0.5, 0.1)] * 10
    result = weak_label_agreement(perfect)
    assert result["accuracy"] == 1.0 and result["kappa"] == pytest.approx(1.0)
    assert result["recorded_share"] == 1.0
    always_merge = [weak_row(0.4, 0.8, 0.0, 0.3), weak_row(0.4, 0.8, 0.5, 0.1)] * 10
    result = weak_label_agreement(always_merge)
    assert result["weak_merge_share"] == 1.0
    assert result["true_merge_share"] == 0.5
    assert result["kappa"] == 0.0
