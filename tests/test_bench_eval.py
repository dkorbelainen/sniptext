"""Benchmark plumbing that needs no OCR: engine runner, legacy baseline, row building."""

import numpy as np
from PIL import Image

from benchmarks import engines
from benchmarks.engines import EngineRunner, RunResult
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
