"""Ensemble OCR - combines multiple engines for better accuracy."""

import numpy as np
from difflib import SequenceMatcher
from collections import Counter
from loguru import logger


class EnsembleOCR:
    """Combine results from multiple OCR engines using voting."""

    def __init__(self):
        """Initialize ensemble."""
        self.results = []

    def combine_results(self, results: list[str]) -> str:
        """
        Combine multiple OCR results using intelligent merging.

        Args:
            results: List of OCR results from different engines

        Returns:
            Combined text with best accuracy
        """
        if not results:
            return ""

        if len(results) == 1:
            return results[0]

        # Filter empty results
        results = [r for r in results if r.strip()]

        if not results:
            return ""

        logger.info(f"Combining {len(results)} OCR results")

        # Split into lines, keep empty lines for structure
        all_lines = [r.split('\n') for r in results]
        max_lines = max(len(lines) for lines in all_lines)

        combined_lines = []

        for line_idx in range(max_lines):
            # Collect all variants of this line
            line_variants = []
            for lines in all_lines:
                if line_idx < len(lines):
                    line_variants.append(lines[line_idx])

            if not line_variants:
                continue

            # Pick best line variant
            best_line = self._pick_best_line(line_variants)
            if best_line.strip():  # Only add non-empty lines
                combined_lines.append(best_line)

        return '\n'.join(combined_lines)

    def _pick_best_line(self, variants: list[str]) -> str:
        """Pick best line variant using multiple heuristics."""
        if len(variants) == 1:
            return variants[0]

        # Strip all variants for comparison
        stripped = [(v.strip(), v) for v in variants]
        variants = [s[0] for s in stripped if s[0]]

        if not variants:
            return ""

        if len(variants) == 1:
            return variants[0]

        # Score each variant
        scores = []
        for v in variants:
            score = 0

            # Length score (longer is usually more complete)
            score += len(v) * 0.1

            # Punctuation score (proper sentences end with punctuation)
            if v.endswith(('.', '!', '?', ':', ',')):
                score += 10

            # Cyrillic letter score (prefer proper Russian text)
            cyrillic_count = sum(1 for c in v if 'а' <= c <= 'я' or 'А' <= c <= 'Я')
            score += cyrillic_count * 0.2

            # Penalize garbage characters at start
            if v and v[0].isdigit():
                score -= 5
            if len(v) > 1 and v[1] == ' ' and v[0].isupper() and len(v.split()[0]) == 1:
                score -= 5  # Single capital letter at start

            scores.append((score, v))

        # Return highest scored variant
        best = max(scores, key=lambda x: x[0])
        logger.debug(f"Selected variant (score {best[0]:.1f}): {best[1][:50]}...")
        return best[1]

    def calculate_confidence(self, results: list[str]) -> float:
        """
        Calculate confidence based on agreement between results.

        Returns:
            Confidence score 0.0-1.0
        """
        if len(results) < 2:
            return 0.5

        # Calculate pairwise similarity
        similarities = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                sim = SequenceMatcher(None, results[i], results[j]).ratio()
                similarities.append(sim)

        if not similarities:
            return 0.5

        avg_similarity = np.mean(similarities)

        logger.debug(f"OCR agreement: {avg_similarity:.2%}")

        return avg_similarity


def post_process_text(text: str) -> str:
    """
    Post-process OCR text - minimal universal cleaning.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    if not text:
        return text

    import re

    # Only universal fixes that work for any language:

    # 1. Fix multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # 2. Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # 3. Fix missing space after punctuation
    text = re.sub(r'([.,!?:;])([а-яА-ЯёЁa-zA-Z])', r'\1 \2', text)

    # 4. Remove empty lines
    lines = [l.strip() for l in text.split('\n')]
    lines = [l for l in lines if l]

    text = '\n'.join(lines)

    logger.debug("Applied universal post-processing")

    return text
