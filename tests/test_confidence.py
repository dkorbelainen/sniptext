"""Tests for ConfidenceModel online learning."""

import json
import numpy as np
import pytest
from pathlib import Path

from sniptext.confidence import ConfidenceModel


@pytest.fixture
def model_in_tmp(tmp_path):
    """ConfidenceModel whose files land in tmp_path."""
    m = ConfidenceModel(model_path=tmp_path / "confidence_model.pkl")
    return m


class TestRecordResult:
    def test_creates_feedback_file(self, model_in_tmp, tmp_path):
        features = np.array([0.5, 0.5, 0.5, 0.0, 0.5])
        model_in_tmp.record_result(features, "fast", True)
        assert model_in_tmp.feedback_path.exists()

    def test_appends_jsonl_line(self, model_in_tmp):
        features = np.array([0.3, 0.4, 0.5, 0.0, 0.6])
        model_in_tmp.record_result(features, "ensemble", False)
        lines = model_in_tmp.feedback_path.read_text().strip().splitlines()
        assert len(lines) == 1
        sample = json.loads(lines[0])
        assert sample["label"] == 1  # ensemble -> 1
        assert sample["success"] is False
        assert len(sample["features"]) == 5

    def test_fast_strategy_label_zero(self, model_in_tmp):
        features = np.array([0.8, 0.7, 0.6, 0.0, 0.9])
        model_in_tmp.record_result(features, "fast", True)
        lines = model_in_tmp.feedback_path.read_text().strip().splitlines()
        sample = json.loads(lines[0])
        assert sample["label"] == 0

    def test_multiple_calls_append(self, model_in_tmp):
        features = np.array([0.5, 0.5, 0.5, 0.0, 0.5])
        for _ in range(5):
            model_in_tmp.record_result(features, "fast", True)
        lines = model_in_tmp.feedback_path.read_text().strip().splitlines()
        assert len(lines) == 5


class TestRetrainFromFeedback:
    def test_retrain_saves_model(self, model_in_tmp):
        """After _retrain_from_feedback with enough samples, model file appears."""
        rng = np.random.default_rng(42)
        features = rng.random((25, 5))
        model_in_tmp.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_in_tmp.feedback_path, "w") as f:
            for i, feat in enumerate(features):
                sample = {"features": feat.tolist(), "label": i % 2, "success": True}
                f.write(json.dumps(sample) + "\n")

        model_in_tmp._retrain_from_feedback()
        # Model file should be created if sklearn is available
        try:
            import sklearn  # noqa: F401

            assert model_in_tmp.model_path.exists()
        except ImportError:
            pytest.skip("sklearn not installed")

    def test_retrain_too_few_samples_no_crash(self, model_in_tmp):
        """Retraining with <10 samples must not crash."""
        model_in_tmp.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        features = np.array([0.5, 0.5, 0.5, 0.0, 0.5])
        with open(model_in_tmp.feedback_path, "w") as f:
            sample = {"features": features.tolist(), "label": 0, "success": True}
            f.write(json.dumps(sample) + "\n")
        # Must complete without exception
        model_in_tmp._retrain_from_feedback()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
