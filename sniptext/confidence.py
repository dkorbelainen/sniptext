"""Confidence scoring model for adaptive OCR selection."""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image

from .analyzer import ImageAnalyzer

# After this many new feedback samples, automatically retrain and save.
_RETRAIN_EVERY = 20


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

    def _ensure_initialized(self):
        """Ensure model is loaded/trained (called on first use)."""
        if self._initialized:
            return

        # Try to load existing model
        self._load_model()

        # Initialize with default model if no trained model exists
        if not self.trained:
            self._initialize_default_model()

        self._initialized = True

    def _initialize_default_model(self):
        """Initialize model with synthetic training data based on heuristics."""
        logger.info("Initializing confidence model with baseline data")

        # Create synthetic training data based on known patterns
        X_train = []
        y_train = []

        # Generate synthetic samples
        # Format: [brightness, contrast, sharpness, has_color, size_ratio]

        # Easy cases (0 = use fast Tesseract only) - 60% of data
        for _ in range(60):
            # High contrast, good brightness, sharp images
            X_train.append(
                [
                    np.random.uniform(0.5, 0.85),  # good brightness
                    np.random.uniform(0.4, 0.9),  # high contrast
                    np.random.uniform(0.5, 1.0),  # sharp
                    np.random.randint(0, 2),  # color doesn't matter
                    np.random.uniform(0.2, 0.9),  # normal ratio
                ]
            )
            y_train.append(0)  # Fast mode

        # Hard cases (1 = use ensemble) - 40% of data
        for _ in range(40):
            # Generate various difficult scenarios
            scenario = np.random.choice(["low_contrast", "extreme_brightness", "blurry"])

            if scenario == "low_contrast":
                X_train.append(
                    [
                        np.random.uniform(0.3, 0.7),  # any brightness
                        np.random.uniform(0.05, 0.25),  # low contrast (key indicator)
                        np.random.uniform(0.2, 0.6),  # moderate sharpness
                        np.random.randint(0, 2),
                        np.random.uniform(0.2, 0.9),
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
                    ]
                )
            else:  # blurry
                X_train.append(
                    [
                        np.random.uniform(0.3, 0.8),
                        np.random.uniform(0.2, 0.5),
                        np.random.uniform(0.05, 0.3),  # low sharpness (key indicator)
                        np.random.randint(0, 2),
                        np.random.uniform(0.2, 0.9),
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

            # Log feature importances
            feature_names = ["brightness", "contrast", "sharpness", "has_color", "size_ratio"]
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
        # Ensure model is initialized on first use
        self._ensure_initialized()

        # Extract features
        features = self.analyzer.extract_features(image)

        contrast = features[1]
        sharpness = features[2]

        # Calculate quality score
        # High contrast and good sharpness = easy case
        quality_score = contrast * 0.6 + sharpness * 0.4

        # Decision rules
        if contrast > 0.5 and sharpness > 0.4:
            # Good quality - use fast mode
            strategy = "fast"
            confidence = min(quality_score, 0.95)
        elif contrast < 0.2 or sharpness < 0.2:
            # Poor quality - definitely use ensemble
            strategy = "ensemble"
            confidence = 0.9
        else:
            # Use trained model for borderline cases
            if not self.trained or self.model is None:
                return self._heuristic_strategy(image)

            features_reshaped = features.reshape(1, -1)
            prediction = self.model.predict(features_reshaped)[0]
            probabilities = self.model.predict_proba(features_reshaped)[0]

            strategy = "fast" if prediction == 0 else "ensemble"
            confidence = probabilities[prediction]

        logger.debug(
            f"Predicted strategy: {strategy} (confidence: {confidence:.2f}, quality: {quality_score:.2f})"
        )

        return strategy, confidence

    def _heuristic_strategy(self, image: Image.Image) -> tuple[str, float]:
        """Fallback heuristic when model is not available."""
        features = self.analyzer.extract_features(image)

        # Simple heuristic: if contrast and sharpness are good, use fast mode
        contrast = features[1]
        sharpness = features[2]

        quality_score = (contrast + sharpness) / 2

        if quality_score > 0.5:
            return "fast", quality_score
        else:
            return "ensemble", 1.0 - quality_score

    def record_result(self, features: np.ndarray, strategy: str, success: bool):
        """
        Record an OCR result for future model improvement.

        Samples are appended to ``ocr_feedback.jsonl`` (one JSON line each).
        Every :data:`_RETRAIN_EVERY` new samples the model is retrained on
        the accumulated feedback and saved to disk so subsequent sessions
        benefit from the learning immediately.

        Args:
            features: Image feature vector (from ImageAnalyzer.extract_features).
            strategy: Strategy that was used — ``'fast'`` or ``'ensemble'``.
            success: ``True`` if OCR produced a non-empty result.
        """
        label = 0 if strategy == "fast" else 1

        sample = {
            "features": features.tolist(),
            "label": label,
            "success": success,
        }

        try:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.feedback_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")
        except Exception as e:
            logger.warning(f"Could not write feedback sample: {e}")
            return

        logger.debug(f"Recorded feedback: strategy={strategy}, success={success}")

        # Count lines to decide if retraining is due
        try:
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                n_samples = sum(1 for _ in f)
        except Exception:
            return

        if n_samples % _RETRAIN_EVERY == 0:
            logger.info(f"Retraining confidence model on {n_samples} feedback samples…")
            self._retrain_from_feedback()

    def _retrain_from_feedback(self):
        """Retrain the confidence model using accumulated feedback data."""
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
            logger.debug("Not enough samples to retrain")
            return

        X = np.array([s["features"] for s in samples])
        y = np.array([s["label"] for s in samples])

        try:
            from sklearn.ensemble import GradientBoostingClassifier

            new_model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
            )
            new_model.fit(X, y)
            self.model = new_model
            self.trained = True
            self.save_model()
            logger.info(f"Confidence model retrained on {len(samples)} samples and saved")
        except ImportError:
            logger.debug("sklearn not available — skipping retrain")
        except Exception as e:
            logger.warning(f"Retrain failed: {e}")

    def _load_model(self):
        """Load trained model from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
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
