"""Ensemble OCR - combines multiple engines for better accuracy."""

from difflib import SequenceMatcher

import numpy as np
from loguru import logger

from .corrector import OCRCorrector


def _reflow(words: list[str], ref_lines: list[str]) -> list[str]:
    """
    Redistribute *words* into lines using *ref_lines* word-count proportions.

    When the two engines split the same content at different points, the
    merged word list needs to be broken back into lines.  We use the longer
    (more granular) source as the template so paragraph structure is kept.
    """
    if not words:
        return []
    if len(ref_lines) <= 1:
        return [" ".join(words)]

    ref_counts = [len(line.split()) for line in ref_lines]
    total_ref = sum(ref_counts) or 1
    total = len(words)

    result: list[str] = []
    idx = 0
    for i, ref_count in enumerate(ref_counts):
        remaining = total - idx
        # Preserve explicit blank lines from ref_lines (ref_count == 0)
        if ref_count == 0:
            chunk = ""
        elif i == len(ref_counts) - 1:
            # Last line: take all remaining words (may be zero)
            chunk = " ".join(words[idx:])
            idx = total
        else:
            # Allocate words proportionally, but not more than remaining
            proportional = round(total * ref_count / total_ref)
            n = min(remaining, max(1, proportional)) if remaining > 0 else 0
            chunk = " ".join(words[idx : idx + n])
            idx += n
        # Append even empty chunks so that blank lines are preserved
        result.append(chunk)
    return result


class EnsembleOCR:
    """Combine results from multiple OCR engines using voting."""

    def __init__(self):
        """Initialize ensemble."""
        self.results = []

    def combine_results(self, results: list[str]) -> str:
        """
        Combine multiple OCR results using intelligent merging.

        Aligns line sequences with SequenceMatcher so that differently-
        segmented output from Tesseract and EasyOCR is handled correctly,
        then resolves word-level disagreements within misaligned blocks.

        Args:
            results: List of OCR results from different engines

        Returns:
            Combined text with best accuracy
        """
        if not results:
            return ""

        results = [r for r in results if r.strip()]

        if not results:
            return ""

        if len(results) == 1:
            return results[0]

        logger.info(f"Combining {len(results)} OCR results")

        merged = results[0]
        for other in results[1:]:
            merged = self._merge_two(merged, other)
        return merged

    def _merge_two(self, a: str, b: str) -> str:
        """Merge two OCR results using line-level then word-level alignment."""
        lines_a = a.splitlines()
        lines_b = b.splitlines()

        matcher = SequenceMatcher(None, lines_a, lines_b, autojunk=False)
        out: list[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                out.extend(lines_a[i1:i2])
            elif tag == "replace":
                out.extend(self._merge_word_level(lines_a[i1:i2], lines_b[j1:j2]))
            elif tag == "delete":
                out.extend(lines_a[i1:i2])
            elif tag == "insert":
                out.extend(lines_b[j1:j2])

        return "\n".join(out)

    def _merge_word_level(self, lines_a: list[str], lines_b: list[str]) -> list[str]:
        """
        Resolve disagreements between two line blocks at word level.

        Joins each block into a word sequence, aligns with SequenceMatcher,
        and picks the better word segment for each differing region.
        The result is reflowed back into lines using the longer block's
        word-count proportions.
        """
        words_a = " ".join(lines_a).split()
        words_b = " ".join(lines_b).split()

        if not words_a:
            return lines_b
        if not words_b:
            return lines_a

        matcher = SequenceMatcher(None, words_a, words_b, autojunk=False)
        merged: list[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                merged.extend(words_a[i1:i2])
            elif tag == "replace":
                seg_a, seg_b = words_a[i1:i2], words_b[j1:j2]
                chosen = seg_a if self._score_words(seg_a) >= self._score_words(seg_b) else seg_b
                merged.extend(chosen)
            elif tag == "delete":
                merged.extend(words_a[i1:i2])
            elif tag == "insert":
                merged.extend(words_b[j1:j2])

        ref = lines_a if len(lines_a) >= len(lines_b) else lines_b
        return _reflow(merged, ref)

    def _score_words(self, words: list[str]) -> float:
        """Score a word sequence: prefer high alphanumeric ratio and completeness."""
        if not words:
            return 0.0
        text = " ".join(words)
        alnum = sum(c.isalnum() for c in text)
        return (alnum / len(text)) * 10 + len(text) * 0.01

    def calculate_confidence(self, results: list[str]) -> float:
        """
        Calculate confidence based on agreement between results.

        Returns:
            Confidence score 0.0-1.0
        """
        if len(results) < 2:
            return 0.5

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


def post_process_text(
    text: str, language: str = "eng", enable_correction: bool = True, aggressive: bool = False
) -> str:
    """
    Post-process OCR text with optional correction.

    Args:
        text: Raw OCR text
        language: Language code
        enable_correction: Apply error corrections
        aggressive: Apply aggressive corrections

    Returns:
        Cleaned text
    """
    if not text:
        return text

    if enable_correction:
        # For multi-language configs (e.g. "eng+rus") detect the dominant
        # script in the recognised text and correct with the right language.
        from .corrector import detect_dominant_language

        candidates = [lang.strip() for lang in language.split("+")]
        effective_lang = detect_dominant_language(text, candidates)
        corrector = OCRCorrector(effective_lang)
        text = corrector.correct(text, aggressive=aggressive)
    else:
        import re

        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?:;])([а-яА-ЯёЁa-zA-Z])", r"\1 \2", text)

        lines = [line.strip() for line in text.split("\n")]
        cleaned: list[str] = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank and cleaned:
                    cleaned.append("")
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False
        text = "\n".join(cleaned).strip()

    return text
