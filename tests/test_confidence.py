"""Tests for ConfidenceModel online learning."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sniptext.confidence import ConfidenceModel


@pytest.fixture
def model_in_tmp(tmp_path):
    """ConfidenceModel whose files land in tmp_path."""
    m = ConfidenceModel(model_path=tmp_path / "confidence_model.pkl")
    return m


class TestRecordResult:
    def test_creates_feedback_file(self, model_in_tmp, tmp_path):
        features = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.15, 0.05])
        model_in_tmp.record_result(features, "fast", True)
        assert model_in_tmp.feedback_path.exists()

    def test_appends_jsonl_line(self, model_in_tmp):
        features = np.array([0.3, 0.4, 0.5, 0.0, 0.6, 0.10, 0.08])
        model_in_tmp.record_result(features, "ensemble", False)
        lines = model_in_tmp.feedback_path.read_text().strip().splitlines()
        assert len(lines) == 1
        sample = json.loads(lines[0])
        assert sample["label"] == 1  # ensemble -> 1
        assert sample["success"] is False
        assert len(sample["features"]) == 7

    def test_fast_strategy_label_zero(self, model_in_tmp):
        features = np.array([0.8, 0.7, 0.6, 0.0, 0.9, 0.20, 0.03])
        model_in_tmp.record_result(features, "fast", True)
        lines = model_in_tmp.feedback_path.read_text().strip().splitlines()
        sample = json.loads(lines[0])
        assert sample["label"] == 0

    def test_multiple_calls_append(self, model_in_tmp):
        features = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.15, 0.05])
        for _ in range(5):
            model_in_tmp.record_result(features, "fast", True)
        lines = model_in_tmp.feedback_path.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_wrong_feature_count_rejected(self, model_in_tmp):
        """Feedback with wrong-length vector must be silently dropped."""
        bad_features = np.array([0.5, 0.5, 0.5, 0.0, 0.5])  # old 5-element format
        model_in_tmp.record_result(bad_features, "fast", True)
        assert not model_in_tmp.feedback_path.exists()


class TestPredictStrategyMLBranch:
    """Tests for the trained-model branch inside predict_strategy().

    The ML path is taken when contrast is in [0.2, 0.5] AND sharpness is in
    [0.2, 0.4] — values that don't match either heuristic shortcut.
    """

    # Borderline features: contrast=0.35, sharpness=0.30 → ML branch
    _BORDERLINE = np.array([0.5, 0.35, 0.30, 0.0, 0.5, 0.15, 0.10])

    def _make_trained_model(self, tmp_path) -> ConfidenceModel:
        m = ConfidenceModel(model_path=tmp_path / "confidence_model.pkl")
        m._initialized = True  # skip lazy init
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])  # fast
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        m.model = mock_model
        m.trained = True
        return m

    def test_ml_model_predict_called_for_borderline_image(self, tmp_path):
        m = self._make_trained_model(tmp_path)
        with patch.object(m.analyzer, "extract_features", return_value=self._BORDERLINE):
            strategy, confidence = m.predict_strategy(MagicMock())
        m.model.predict.assert_called_once()
        m.model.predict_proba.assert_called_once()

    def test_ml_model_returns_fast_when_prediction_is_zero(self, tmp_path):
        m = self._make_trained_model(tmp_path)
        m.model.predict.return_value = np.array([0])
        m.model.predict_proba.return_value = np.array([[0.85, 0.15]])
        with patch.object(m.analyzer, "extract_features", return_value=self._BORDERLINE):
            strategy, confidence = m.predict_strategy(MagicMock())
        assert strategy == "fast"
        assert confidence == pytest.approx(0.85)

    def test_ml_model_returns_ensemble_when_prediction_is_one(self, tmp_path):
        m = self._make_trained_model(tmp_path)
        m.model.predict.return_value = np.array([1])
        m.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        with patch.object(m.analyzer, "extract_features", return_value=self._BORDERLINE):
            strategy, confidence = m.predict_strategy(MagicMock())
        assert strategy == "ensemble"
        assert confidence == pytest.approx(0.8)

    def test_falls_back_to_heuristic_when_not_trained(self, tmp_path):
        m = ConfidenceModel(model_path=tmp_path / "confidence_model.pkl")
        m._initialized = True
        m.trained = False
        m.model = None
        with patch.object(m.analyzer, "extract_features", return_value=self._BORDERLINE):
            strategy, confidence = m.predict_strategy(MagicMock())
        # Heuristic: quality = (contrast + sharpness)/2 = (0.35+0.30)/2 = 0.325 < 0.5 → ensemble
        assert strategy == "ensemble"
        assert isinstance(confidence, float)


class TestRetrainFromFeedback:
    def test_retrain_saves_model(self, model_in_tmp):
        """After _retrain_from_feedback with enough samples, model file appears."""
        rng = np.random.default_rng(42)
        features = rng.random((25, 7))
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
        features = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.15, 0.05])
        with open(model_in_tmp.feedback_path, "w") as f:
            sample = {"features": features.tolist(), "label": 0, "success": True}
            f.write(json.dumps(sample) + "\n")
        # Must complete without exception
        model_in_tmp._retrain_from_feedback()

    def test_retrain_single_class_skips_cv(self, model_in_tmp):
        """When all samples share one label, sklearn can't fit — must not raise."""
        rng = np.random.default_rng(0)
        model_in_tmp.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_in_tmp.feedback_path, "w") as f:
            for feat in rng.random((15, 7)):
                f.write(json.dumps({"features": feat.tolist(), "label": 0, "success": True}) + "\n")
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")
        # Must complete without propagating an exception; training will fail
        # gracefully (sklearn rejects single-class datasets).
        model_in_tmp._retrain_from_feedback()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
