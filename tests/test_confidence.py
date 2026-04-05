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
        assert sample["winner"] == "ensemble"
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
        """Feedback with wrong-length vector must be dropped, with a warning logged."""
        bad_features = np.array([0.5, 0.5, 0.5, 0.0, 0.5])  # old 5-element format
        model_in_tmp.record_result(bad_features, "fast", True)
        assert not model_in_tmp.feedback_path.exists()


class TestPredictStrategyThresholds:
    """Test the hard-threshold branches that bypass the ML model."""

    def _model(self, tmp_path):
        m = ConfidenceModel(model_path=tmp_path / "m.pkl")
        m._initialized = True
        m.trained = False
        m.model = None
        return m

    def test_high_contrast_sharp_low_noise_is_fast(self, tmp_path):
        m = self._model(tmp_path)
        # contrast=0.7, sharpness=0.6, noise=0.1 → fast threshold
        feat = np.array([0.7, 0.7, 0.6, 0.0, 0.5, 0.15, 0.1])
        with patch.object(m.analyzer, "extract_features", return_value=feat):
            strategy, _ = m.predict_strategy(MagicMock())
        assert strategy == "fast"

    def test_very_noisy_image_triggers_ensemble(self, tmp_path):
        m = self._model(tmp_path)
        # noise=0.8 → ensemble regardless of contrast/sharpness
        feat = np.array([0.6, 0.6, 0.5, 0.0, 0.5, 0.15, 0.8])
        with patch.object(m.analyzer, "extract_features", return_value=feat):
            strategy, _ = m.predict_strategy(MagicMock())
        assert strategy == "ensemble"

    def test_near_empty_image_triggers_ensemble(self, tmp_path):
        m = self._model(tmp_path)
        # text_density=0.005 (almost no dark pixels) → ensemble
        feat = np.array([0.95, 0.6, 0.5, 0.0, 0.5, 0.005, 0.05])
        with patch.object(m.analyzer, "extract_features", return_value=feat):
            strategy, _ = m.predict_strategy(MagicMock())
        assert strategy == "ensemble"

    def test_noisy_image_not_incorrectly_labelled_fast(self, tmp_path):
        m = self._model(tmp_path)
        # Old logic: contrast>0.5 AND sharpness>0.4 → fast, ignoring noise.
        # New logic: noise=0.5 blocks the fast path.
        feat = np.array([0.7, 0.6, 0.5, 0.0, 0.5, 0.15, 0.5])
        with patch.object(m.analyzer, "extract_features", return_value=feat):
            strategy, _ = m.predict_strategy(MagicMock())
        # noise=0.5 ≥ 0.4 → should NOT be fast
        assert strategy != "fast"


class TestPredictStrategyMLBranch:
    """Tests for the trained-model branch inside predict_strategy().

    The ML path is taken whenever none of the hard-threshold shortcuts fire:
    the image is neither clearly suitable for the fast path nor clearly bad
    enough to force the ensemble. This includes medium-quality cases as well
    as high-contrast images that are too noisy (e.g. noise >= 0.4) or not
    sharp enough (e.g. sharpness <= 0.4) to satisfy the fast-path criteria.
    """

    # Borderline features: contrast=0.35, sharpness=0.30, noise=0.10 → ML branch
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
        # Heuristic: quality = contrast*0.5 + sharpness*0.3 - noise*0.2
        # = 0.35*0.5 + 0.30*0.3 - 0.10*0.2 = 0.175 + 0.09 - 0.02 = 0.245 < 0.35 → ensemble
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


class TestSaveLoadModel:
    def test_save_load_round_trip(self, model_in_tmp):
        """save_model() then _load_model() must restore trained=True."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")
        model_in_tmp._ensure_initialized()
        assert model_in_tmp.trained

        model_in_tmp.save_model()
        assert model_in_tmp.model_path.exists()

        m2 = ConfidenceModel(model_path=model_in_tmp.model_path)
        m2._load_model()
        assert m2.trained
        assert m2.model is not None

    def test_load_stale_model_is_discarded(self, tmp_path):
        """A model trained on a different feature count must not be loaded."""
        import pickle

        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            pytest.skip("sklearn not installed")

        from sniptext.confidence import _FEATURE_COUNT

        clf = GradientBoostingClassifier()
        clf.n_features_in_ = _FEATURE_COUNT + 1  # wrong count — never fitted

        model_path = tmp_path / "stale.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": clf}, f)

        m = ConfidenceModel(model_path=model_path)
        m._load_model()
        assert not m.trained
        assert m.model is None

    def test_save_without_trained_model_does_not_create_file(self, model_in_tmp):
        """save_model() before training must not create a file."""
        model_in_tmp.save_model()
        assert not model_in_tmp.model_path.exists()

    def test_load_corrupt_pickle_sets_trained_false(self, tmp_path):
        """_load_model() on a corrupt file must not raise and must leave trained=False."""
        model_path = tmp_path / "corrupt.pkl"
        model_path.write_bytes(b"not a valid pickle")

        m = ConfidenceModel(model_path=model_path)
        m._load_model()

        assert not m.trained
        assert m.model is None

    def test_save_model_handles_write_error(self, tmp_path):
        """save_model() must not raise when the file cannot be written."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")

        m = ConfidenceModel(model_path=tmp_path / "model.pkl")
        m._ensure_initialized()
        assert m.trained

        with patch("sniptext.confidence.open", side_effect=OSError("disk full")):
            m.save_model()  # must not raise

        assert not (tmp_path / "model.pkl").exists()


class TestColdStart:
    """Test cold start scenarios (first run with no training data)."""

    def test_cold_start_initializes_synthetic_baseline(self, model_in_tmp):
        """On first use, model should initialize with synthetic data."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")

        model_in_tmp._ensure_initialized()
        assert model_in_tmp.trained
        assert model_in_tmp.model is not None

    def test_cold_start_with_a_b_test(self, model_in_tmp):
        """A/B test should work immediately with synthetic baseline."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")

        model_in_tmp._ensure_initialized()

        # Simulate A/B test recording
        features = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.15, 0.05])
        fast_result = {"text": "test", "quality_score": 0.8, "length": 4}
        ensemble_result = {"text": "test", "quality_score": 0.7, "length": 4}

        model_in_tmp.record_result(
            features, fast_result=fast_result, ensemble_result=ensemble_result
        )

        assert model_in_tmp.feedback_path.exists()
        lines = model_in_tmp.feedback_path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_cold_start_retrain_at_20_samples(self, model_in_tmp):
        """Model should retrain after 20 feedback samples."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")

        model_in_tmp._ensure_initialized()

        # Record 20 A/B test results
        for i in range(20):
            features = np.array([0.5 + i * 0.01, 0.5, 0.5, 0.0, 0.5, 0.15, 0.05])
            if i % 2 == 0:
                fast_result = {"text": "test", "quality_score": 0.8, "length": 4}
                ensemble_result = {"text": "test", "quality_score": 0.7, "length": 4}
            else:
                fast_result = {"text": "test", "quality_score": 0.7, "length": 4}
                ensemble_result = {"text": "test", "quality_score": 0.8, "length": 4}

            model_in_tmp.record_result(
                features, fast_result=fast_result, ensemble_result=ensemble_result
            )

        # At 20 samples, should trigger retrain
        assert model_in_tmp.metrics_history_path.exists()
        metrics = json.loads(model_in_tmp.metrics_history_path.read_text().strip())
        assert metrics["n_samples"] == 20

    def test_migrate_feedback_features_pads_missing(self, model_in_tmp):
        """Feature migration should pad short feature vectors."""
        old_samples = [
            {"features": [0.5, 0.5], "label": 0, "winner": "fast"},  # 2 features
            {"features": [0.3, 0.4, 0.6], "label": 1, "winner": "ensemble"},  # 3 features
        ]

        migrated = model_in_tmp._migrate_feedback_features(old_samples)

        assert len(migrated) == 2
        assert len(migrated[0]["features"]) == 7
        assert len(migrated[1]["features"]) == 7
        assert migrated[0]["_migrated"] is True

    def test_migrate_feedback_features_truncates_extra(self, model_in_tmp):
        """Feature migration should truncate long feature vectors."""
        old_samples = [
            {
                "features": [0.5, 0.5, 0.5, 0.0, 0.5, 0.15, 0.05, 0.1, 0.2],
                "label": 0,
                "winner": "fast",
            },
        ]

        migrated = model_in_tmp._migrate_feedback_features(old_samples)

        assert len(migrated) == 1
        assert len(migrated[0]["features"]) == 7

    def test_migrate_feedback_drops_empty_features(self, model_in_tmp):
        """Feature migration should drop samples with empty features."""
        old_samples = [
            {"features": [0.5, 0.5], "label": 0, "winner": "fast"},
            {"features": [], "label": 1, "winner": "ensemble"},  # Empty
            {"features": [0.3, 0.4, 0.6], "label": 0, "winner": "fast"},
        ]

        migrated = model_in_tmp._migrate_feedback_features(old_samples)

        assert len(migrated) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
