"""Confidence scoring model for adaptive OCR selection."""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger
from PIL import Image

from .analyzer import ImageAnalyzer

# After this many new feedback samples, automatically retrain and save.
_RETRAIN_EVERY = 20

# Expected feature-vector length produced by ImageAnalyzer.extract_features().
# Bump this when new features are added; stale models/feedback are discarded.
_FEATURE_COUNT = 7

class ConfidenceModel:
    """Model to predict OCR confidence and choose optimal strategy."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize confidence model (lazy initialization)."""
        self.analyzer = ImageAnalyzer()
        self.model = None
        self.trained = False
        self._initialized = False

        if model_path is None:
            config_dir = Path.home() / ".config" / "sniptext"
            model_path = config_dir / "confidence_model.pkl"

        self.model_path = Path(model_path)
        self.feedback_path = self.model_path.with_name("ocr_feedback.jsonl")
        self.metrics_history_path = self.model_path.with_name("metrics_history.jsonl")

    def _ensure_initialized(self):
        """Ensure model is loaded/trained (called on first use)."""
        if self._initialized:
            return

        self._load_model()

        if not self.trained:
            self._initialize_default_model()

        self._initialized = True

    def _initialize_default_model(self):
        """Initialize model with synthetic training data based on heuristics."""
        logger.info("Initializing confidence model with baseline data")

        # Create synthetic training data based on known patterns
        # Format: [brightness, contrast, sharpness, has_color, size_ratio,
        #          text_density, noise_level]
        X_train = []
        y_train = []

        # Easy cases (0 = use fast Tesseract only) - 60% of data
        for _ in range(60):
            # High contrast, good brightness, sharp, low noise
            X_train.append(
                [
                    np.random.uniform(0.5, 0.85),  # good brightness
                    np.random.uniform(0.4, 0.9),  # high contrast
                    np.random.uniform(0.5, 1.0),  # sharp
                    np.random.randint(0, 2),  # color doesn't matter
                    np.random.uniform(0.2, 0.9),  # normal ratio
                    np.random.uniform(0.05, 0.30),  # normal text density
                    np.random.uniform(0.0, 0.15),  # low noise
                ]
            )
            y_train.append(0)  # Fast mode

        # Hard cases (1 = use ensemble) - 40% of data
        for _ in range(40):
            # Generate various difficult scenarios
            scenario = np.random.choice(["low_contrast", "extreme_brightness", "blurry", "noisy"])

            if scenario == "low_contrast":
                X_train.append(
                    [
                        np.random.uniform(0.3, 0.7),  # any brightness
                        np.random.uniform(0.05, 0.25),  # low contrast (key indicator)
                        np.random.uniform(0.2, 0.6),  # moderate sharpness
                        np.random.randint(0, 2),
                        np.random.uniform(0.2, 0.9),
                        np.random.uniform(0.01, 0.4),  # sparse or moderate text
                        np.random.uniform(0.1, 0.5),  # moderate noise
                    ]
                )
            elif scenario == "extreme_brightness":
                X_train.append(
                    [
                        np.random.choice(
                            [np.random.uniform(0.05, 0.25), np.random.uniform(0.85, 1.0)]
                        ),  # very dark or bright
                        np.random.uniform(0.15, 0.4),  # lower contrast
                        np.random.uniform(0.3, 0.7),
                        np.random.randint(0, 2),
                        np.random.uniform(0.2, 0.9),
                        np.random.uniform(0.01, 0.5),
                        np.random.uniform(0.05, 0.4),
                    ]
                )
            elif scenario == "blurry":
                X_train.append(
                    [
                        np.random.uniform(0.3, 0.8),
                        np.random.uniform(0.2, 0.5),
                        np.random.uniform(0.05, 0.3),  # low sharpness (key indicator)
                        np.random.randint(0, 2),
                        np.random.uniform(0.2, 0.9),
                        np.random.uniform(0.01, 0.3),
                        np.random.uniform(0.05, 0.35),
                    ]
                )
            else:  # noisy
                X_train.append(
                    [
                        np.random.uniform(0.3, 0.8),
                        np.random.uniform(0.15, 0.45),
                        np.random.uniform(0.2, 0.6),
                        np.random.randint(0, 2),
                        np.random.uniform(0.2, 0.9),
                        np.random.uniform(0.01, 0.15),  # sparse text density
                        np.random.uniform(0.35, 0.9),  # high noise (key indicator)
                    ]
                )

            y_train.append(1)  # Ensemble mode

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train model (lazy import sklearn to speed up startup)
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            )
            self.model.fit(X_train, y_train)
            self.trained = True

            feature_names = [
                "brightness",
                "contrast",
                "sharpness",
                "has_color",
                "size_ratio",
                "text_density",
                "noise_level",
            ]
            importances = self.model.feature_importances_
            logger.debug(f"Feature importances: {dict(zip(feature_names, importances))}")

            logger.info("Baseline confidence model initialized")
        except ImportError:
            logger.warning("scikit-learn not available, using heuristic-only mode")
            self.trained = False

    def predict_strategy(self, image: Image.Image) -> tuple[str, float]:
        """
        Predict optimal OCR strategy for given image.

        Args:
            image: PIL Image to analyze

        Returns:
            Tuple of (strategy, confidence)
            strategy: 'fast' or 'ensemble'
            confidence: probability score (0-1)
        """
        self._ensure_initialized()

        features = self.analyzer.extract_features(image)

        contrast = features[1]
        sharpness = features[2]
        noise = features[6]
        density = features[5]

        # Weighted quality score across three features (contrast, sharpness, noise)
        quality_score = contrast * 0.5 + sharpness * 0.3 - noise * 0.2

        # Clearly bad: low contrast, blurry, very noisy, or almost no text → ensemble
        if contrast < 0.2 or sharpness < 0.2 or noise > 0.6 or density < 0.02:
            strategy = "ensemble"
            confidence = 0.9
        # Clearly good: high contrast, sharp, low noise → fast
        elif contrast > 0.5 and sharpness > 0.4 and noise < 0.4:
            strategy = "fast"
            confidence = min(quality_score + 0.2, 0.95)
        else:
            # Use trained model for borderline cases
            if not self.trained or self.model is None:
                return self._heuristic_strategy(features)

            features_reshaped = features.reshape(1, -1)
            prediction = self.model.predict(features_reshaped)[0]
            probabilities = self.model.predict_proba(features_reshaped)[0]

            strategy = "fast" if prediction == 0 else "ensemble"
            confidence = probabilities[prediction]

        logger.debug(
            f"Predicted strategy: {strategy} (confidence: {confidence:.2f}, "
            f"contrast={contrast:.2f}, sharpness={sharpness:.2f}, "
            f"noise={noise:.2f}, density={density:.2f})"
        )

        return strategy, confidence

    def _heuristic_strategy(self, features: np.ndarray) -> tuple[str, float]:
        """Fallback heuristic when model is not available."""
        contrast = features[1]
        sharpness = features[2]
        noise = features[6]

        # Weighted quality across the three most relevant features
        quality_score = contrast * 0.5 + sharpness * 0.3 - noise * 0.2

        if quality_score > 0.35:
            return "fast", min(quality_score + 0.2, 0.95)
        else:
            return "ensemble", min(1.0 - quality_score, 0.95)

    def record_result(
        self,
        features: np.ndarray,
        strategy: Optional[str] = None,
        success: Optional[bool] = None,
        fast_result: Optional[Dict] = None,
        ensemble_result: Optional[Dict] = None,
    ):
        """
        Record OCR result(s) for model training.

        For A/B testing: provide both fast_result and ensemble_result as dicts.
        The better one (by quality_score) becomes the training label.

        For regular use: provide only the result that was actually used as dict,
        or use legacy parameters (strategy, success) for backward compatibility.

        Samples are appended to ``ocr_feedback.jsonl`` (one JSON line each).
        Every :data:`_RETRAIN_EVERY` new samples the model is retrained on
        the accumulated feedback and saved to disk.

        Args:
            features: Image feature vector (from ImageAnalyzer.extract_features).
            strategy: (Legacy) Strategy name - 'fast' or 'ensemble'
            success: (Legacy) Whether OCR produced non-empty result
            fast_result: Dict with keys 'text', 'quality_score', optional 'confidence'
            ensemble_result: Dict with keys 'text', 'quality_score', optional 'confidence'
        """
        # Handle legacy API
        if strategy is not None and success is not None:
            # Convert legacy call to new format
            if strategy == "fast":
                fast_result = {"text": "", "quality_score": 0.5 if success else 0.0, "length": 0}
            else:
                ensemble_result = {
                    "text": "",
                    "quality_score": 0.5 if success else 0.0,
                    "length": 0,
                }

        if len(features) != _FEATURE_COUNT:
            logger.warning(
                f"Skipping feedback: expected {_FEATURE_COUNT} features, got {len(features)}"
            )
            return

        # Determine label based on which strategy was better
        winner = None
        label = None

        if fast_result is not None and ensemble_result is not None:
            # A/B test: compare both results
            fast_quality = fast_result.get("quality_score", 0.0)
            ensemble_quality = ensemble_result.get("quality_score", 0.0)

            # Only record if there's a clear winner (>2% difference)
            # Lowered threshold from 5% to capture more training data
            if abs(fast_quality - ensemble_quality) >= 0.02:
                if fast_quality > ensemble_quality:
                    winner = "fast"
                    label = 0
                else:
                    winner = "ensemble"
                    label = 1

                logger.debug(
                    f"A/B test: {winner} won (fast={fast_quality:.2f}, "
                    f"ensemble={ensemble_quality:.2f})"
                )
            else:
                logger.debug(
                    f"A/B test: tie (fast={fast_quality:.2f}, "
                    f"ensemble={ensemble_quality:.2f}), not recording"
                )
                return
        elif fast_result is not None:
            # Only fast was used
            winner = "fast"
            label = 0
        elif ensemble_result is not None:
            # Only ensemble was used
            winner = "ensemble"
            label = 1
        else:
            logger.warning("No results provided to record_result")
            return

        sample = {
            "features": features.tolist(),
            "label": label,
            "winner": winner,
            "fast_result": fast_result,
            "ensemble_result": ensemble_result,
        }

        try:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.feedback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
        except Exception as e:
            logger.warning(f"Could not write feedback sample: {e}")
            return

        logger.debug(f"Recorded feedback: winner={winner}")

        # Count lines to decide if retraining is due
        try:
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                n_samples = sum(1 for _ in f)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Could not count feedback samples in {self.feedback_path}: {e}")
            return

        if n_samples % _RETRAIN_EVERY == 0:
            logger.info(f"Retraining confidence model on {n_samples} feedback samples…")
            self._retrain_from_feedback()

    def _retrain_from_feedback(self):
        """Retrain the confidence model using accumulated feedback data with metrics tracking."""
        try:
            samples = []
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Could not read feedback file: {e}")
            return

        if len(samples) < 10:
            logger.warning(f"Could not retrain: only {len(samples)} samples (need ≥10)")
            return

        # Try to migrate samples with mismatched feature count
        samples = self._migrate_feedback_features(samples)

        if len(samples) < 10:
            logger.warning(
                f"Could not retrain after migration: only {len(samples)} valid samples (need ≥10)"
            )
            return

        X = np.array([s["features"] for s in samples])
        y = np.array([s["label"] for s in samples])

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.model_selection import cross_val_score, train_test_split

            # Split data for validation (80/20)
            test_size = max(0.2, min(10, len(samples) * 0.2) / len(samples))
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            # Warn if validation set is too small (< 5 samples = unreliable metrics)
            if len(X_val) < 5:
                logger.warning(
                    f"Validation set very small ({len(X_val)} samples); "
                    "accuracy metrics may be unreliable — collect more feedback"
                )

            new_model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            )
            new_model.fit(X_train, y_train)

            # Calculate metrics
            train_acc = accuracy_score(y_train, new_model.predict(X_train))
            val_acc = accuracy_score(y_val, new_model.predict(X_val)) if len(X_val) > 0 else 0.0

            # Get predictions for validation set
            if len(X_val) > 0:
                y_pred = new_model.predict(X_val)
                conf_matrix = confusion_matrix(y_val, y_pred)

                # Calculate per-class metrics
                class_report = classification_report(
                    y_val,
                    y_pred,
                    target_names=["fast", "ensemble"],
                    output_dict=True,
                    zero_division=0,
                )
            else:
                conf_matrix = None
                class_report = None

            # Save model
            self.model = new_model
            self.trained = True
            self.save_model()

            logger.info(
                f"Model retrained on {len(samples)} samples | "
                f"Train acc: {train_acc:.1%} | Val acc: {val_acc:.1%}"
            )

            # Log detailed metrics
            if class_report:
                fast_metrics = class_report.get("fast", {})
                ensemble_metrics = class_report.get("ensemble", {})
                logger.info(
                    f"Fast strategy - Precision: {fast_metrics.get('precision', 0):.2f}, "
                    f"Recall: {fast_metrics.get('recall', 0):.2f}, "
                    f"F1: {fast_metrics.get('f1-score', 0):.2f}"
                )
                logger.info(
                    f"Ensemble strategy - Precision: {ensemble_metrics.get('precision', 0):.2f}, "
                    f"Recall: {ensemble_metrics.get('recall', 0):.2f}, "
                    f"F1: {ensemble_metrics.get('f1-score', 0):.2f}"
                )

            # Feature importance
            feature_names = [
                "brightness",
                "contrast",
                "sharpness",
                "has_color",
                "size_ratio",
                "text_density",
                "noise_level",
            ]
            importances = new_model.feature_importances_
            top_features = sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            )[:3]
            logger.info(
                f"Top features: {', '.join(f'{name}={imp:.2f}' for name, imp in top_features)}"
            )

            # Save metrics to history file
            self._save_metrics_history(
                {
                    "n_samples": len(samples),
                    "train_accuracy": float(train_acc),
                    "val_accuracy": float(val_acc),
                    "confusion_matrix": conf_matrix.tolist() if conf_matrix is not None else None,
                    "class_report": class_report,
                    "feature_importance": dict(zip(feature_names, importances.tolist())),
                    "label_distribution": {
                        "fast": int(np.sum(y == 0)),
                        "ensemble": int(np.sum(y == 1)),
                    },
                }
            )

        except ImportError:
            logger.debug("sklearn not available — skipping retrain")
        except Exception as e:
            logger.warning(f"Retrain failed: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return

        # Cross-validation for additional validation
        try:
            from sklearn.model_selection import cross_val_score

            _, counts = np.unique(y, return_counts=True)
            n_folds = min(5, int(counts.min()))
            if n_folds >= 2:
                cv_scores = cross_val_score(self.model, X, y, cv=n_folds, scoring="accuracy")
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
                if cv_mean < 0.65:
                    logger.warning(
                        f"Low cross-validation accuracy ({cv_mean:.1%} ±{cv_std:.1%}); "
                        "model may not generalise well — collect more feedback"
                    )
                else:
                    logger.info(f"CV accuracy: {cv_mean:.1%} ±{cv_std:.1%}")
            else:
                logger.info(
                    f"Cross-validation skipped: minority class has {int(counts.min())} "
                    f"samples (need ≥2 per class for {n_folds}-fold CV)"
                )
        except Exception as e:
            logger.debug(f"CV evaluation skipped: {e}")

    def _migrate_feedback_features(self, samples: list) -> list:
        """
        Migrate feedback samples when feature count changes.

        If a sample has fewer features than current _FEATURE_COUNT,
        pad it with average values from other samples (or defaults).
        If a sample has more features, truncate it.

        Args:
            samples: List of feedback sample dicts (with 'features' key)

        Returns:
            List of migrated samples with correct feature count
        """
        migrated = []
        dropped = 0

        for sample in samples:
            old_features = sample.get("features", [])
            old_count = len(old_features)

            if old_count == _FEATURE_COUNT:
                # Already correct size
                migrated.append(sample)
            elif old_count > 0:
                # Try to migrate: pad or truncate
                if old_count < _FEATURE_COUNT:
                    # Pad with average of provided features or 0.5 default
                    avg_val = sum(old_features) / len(old_features) if old_features else 0.5
                    new_features = old_features + [avg_val] * (_FEATURE_COUNT - old_count)
                else:
                    # Truncate
                    new_features = old_features[:_FEATURE_COUNT]

                sample["features"] = new_features
                sample["_migrated"] = True
                migrated.append(sample)
            else:
                # Empty features, can't migrate
                dropped += 1

        if dropped > 0:
            logger.warning(f"Dropped {dropped} samples with empty feature vectors")

        if len(migrated) < len(samples):
            logger.info(
                f"Migrated feedback: {len(migrated)} samples with adjusted features "
                f"(from {samples[0].get('_old_feature_count', old_count)} to {_FEATURE_COUNT})"
            )

        return migrated

    def _load_model(self):
        """Load trained model from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    model = data["model"]
                # Discard models trained on a different feature count to avoid
                # shape-mismatch errors when the feature vector was extended.
                actual = getattr(model, "n_features_in_", None)
                if actual is not None and actual != _FEATURE_COUNT:
                    logger.warning(
                        f"Discarding stale model (trained on {actual} features, "
                        f"current={_FEATURE_COUNT}). Will retrain."
                    )
                    return
                self.model = model
                self.trained = True
                logger.info(f"Loaded confidence model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.trained = False

    def save_model(self):
        """Save trained model to disk."""
        if not self.trained:
            logger.warning("No trained model to save")
            return

        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump({"model": self.model}, f)
            logger.info(f"Saved confidence model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _save_metrics_history(self, metrics: Dict):
        """
        Save training metrics to history file for tracking model improvement.

        Args:
            metrics: Dictionary containing training metrics
        """
        from datetime import datetime, timezone

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }

        try:
            self.metrics_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug(f"Saved metrics to {self.metrics_history_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics history: {e}")
