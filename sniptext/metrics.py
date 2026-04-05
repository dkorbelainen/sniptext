"""Quality metrics for OCR results evaluation."""

import re
from typing import Optional

import numpy as np


class OCRQualityMetrics:
    """Calculate quality scores for OCR results."""

    def __init__(self):
        """Initialize quality metrics calculator."""
        self.spell_checker = None
        self._try_init_spell_checker()

    def _try_init_spell_checker(self):
        """Try to initialize spell checker if available."""
        try:
            from symspellpy import SymSpell

            self.spell_checker = SymSpell(max_dictionary_edit_distance=2)
        except ImportError:
            pass

    def calculate_quality_score(
        self,
        text: str,
        confidence_scores: Optional[list[float]] = None,
    ) -> float:
        """
        Calculate overall quality score for OCR result.

        Combines multiple indicators into a single normalized score (0-1).
        Higher score = better quality.

        Args:
            text: Recognized text
            confidence_scores: Optional list of per-character confidence values (0-100)

        Returns:
            Quality score between 0 and 1
        """
        if not text:
            return 0.0

        scores = []
        weights = []

        # 1. Average confidence (if available) - weight 0.35
        if confidence_scores:
            avg_conf = np.mean(confidence_scores) / 100.0  # normalize to 0-1
            scores.append(avg_conf)
            weights.append(0.35)

        # 2. Text length score - weight 0.15
        # Normalize: 1-10 chars = low, 20+ = good
        length_score = min(len(text) / 20.0, 1.0)
        scores.append(length_score)
        weights.append(0.15)

        # 3. Alphanumeric ratio - weight 0.20
        # Higher ratio of normal chars vs special/garbage
        alnum_ratio = self._calculate_alnum_ratio(text)
        scores.append(alnum_ratio)
        weights.append(0.20)

        # 4. Suspicious patterns detection - weight 0.15
        # Detect garbage: ���, repeated chars, etc.
        suspicious_score = 1.0 - self._detect_suspicious_patterns(text)
        scores.append(suspicious_score)
        weights.append(0.15)

        # 5. Dictionary words ratio (if spell checker available) - weight 0.15
        if self.spell_checker is not None:
            dict_score = self._calculate_dictionary_ratio(text)
            scores.append(dict_score)
            weights.append(0.15)

        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            quality = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            quality = 0.0

        return float(np.clip(quality, 0.0, 1.0))

    def _calculate_alnum_ratio(self, text: str) -> float:
        """Calculate ratio of alphanumeric + common punctuation to total."""
        if not text:
            return 0.0

        # Count valid chars: letters, digits, common punctuation, whitespace
        valid_chars = sum(
            1
            for c in text
            if c.isalnum() or c in " .,!?;:'\"-()[]{}@#$%&*+=/<>\\|`~\n\t"
        )
        return valid_chars / len(text)

    def _detect_suspicious_patterns(self, text: str) -> float:
        """
        Detect suspicious patterns indicating OCR errors.

        Returns:
            Suspiciousness score between 0 (clean) and 1 (very suspicious)
        """
        if not text:
            return 1.0

        issues = 0

        # Check for replacement characters (garbage)
        if "�" in text or "\ufffd" in text:
            issues += 3

        # Check for excessive repeated characters (more than 4 in a row)
        if re.search(r"(.)\1{4,}", text):
            issues += 2

        # Check for very short "words" with many special chars
        words = text.split()
        if words:
            special_char_words = sum(1 for w in words if len(w) <= 3 and not w.isalnum())
            if special_char_words / len(words) > 0.3:
                issues += 2

        # Check for excessive special characters overall
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(
            text
        )
        if special_ratio > 0.4:
            issues += 1

        # Normalize to 0-1 range (max issues ~8, clamp at 5)
        return min(issues / 5.0, 1.0)

    def _calculate_dictionary_ratio(self, text: str) -> float:
        """Calculate ratio of valid dictionary words to total words."""
        if not self.spell_checker:
            return 0.5  # neutral when unavailable

        words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
        if not words:
            return 0.0

        valid_words = 0
        for word in words:
            # Check if word is in dictionary (allowing small edit distance)
            suggestions = self.spell_checker.lookup(
                word.lower(), verbosity=0, max_edit_distance=1
            )
            if suggestions:
                valid_words += 1

        return valid_words / len(words) if words else 0.0

    def compare_results(
        self,
        result1: dict,
        result2: dict,
    ) -> str:
        """
        Compare two OCR results and determine which is better.

        Args:
            result1: Dict with 'text', 'quality_score', optionally 'confidence'
            result2: Dict with 'text', 'quality_score', optionally 'confidence'

        Returns:
            'result1', 'result2', or 'tie'
        """
        score1 = result1.get("quality_score", 0.0)
        score2 = result2.get("quality_score", 0.0)

        # Consider tie if scores are very close (within 5%)
        if abs(score1 - score2) < 0.05:
            return "tie"

        return "result1" if score1 > score2 else "result2"


def extract_confidence_scores(tesseract_data: dict) -> Optional[list[float]]:
    """
    Extract per-character confidence scores from Tesseract data.

    Args:
        tesseract_data: Dict from pytesseract.image_to_data(output_type=Output.DICT)

    Returns:
        List of confidence scores (0-100) or None if not available
    """
    if not tesseract_data or "conf" not in tesseract_data:
        return None

    # Filter out -1 confidence (no text detected)
    confidences = [float(c) for c in tesseract_data["conf"] if c != -1]

    return confidences if confidences else None
