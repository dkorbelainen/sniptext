"""Ensemble OCR - combines multiple engines for better accuracy."""

from __future__ import annotations

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

    def combine_results(
        self,
        results: list[str],
        confidences: list[list[list[float]] | None] | None = None,
    ) -> str:
        """
        Combine multiple OCR results using intelligent merging.

        Aligns line sequences with SequenceMatcher so that differently-
        segmented output from Tesseract and EasyOCR is handled correctly,
        then resolves word-level disagreements within misaligned blocks.

        When ``confidences`` is given, ``confidences[k]`` holds per-line word
        confidences (``list[list[float]]`` in ``[0,1]``) aligned to
        ``results[k].splitlines()``; word-level disagreements are then resolved
        by mean segment confidence, falling back to the text heuristic when
        confidence is missing. ``confidences=None`` reproduces the legacy
        behaviour exactly. Only the first merged pair is confidence-aware.

        Args:
            results: OCR results from different engines.
            confidences: Optional per-result, per-line word confidences.

        Returns:
            Combined text with best accuracy.
        """
        if not results:
            return ""

        if confidences is None:
            confidences = [None] * len(results)

        pairs = [(r, c) for r, c in zip(results, confidences) if r.strip()]
        if not pairs:
            return ""
        if len(pairs) == 1:
            return pairs[0][0]

        logger.info(f"Combining {len(pairs)} OCR results")

        pairs.sort(key=lambda p: self._doc_score(p[0], p[1]), reverse=True)

        merged, merged_conf = pairs[0]
        for i, (other, other_conf) in enumerate(pairs[1:]):
            # The merged text has no aligned confidence, so only the first pair
            # is confidence-aware; later pairs fall back to the text heuristic.
            conf_b = other_conf if i == 0 else None
            merged = self._merge_two(merged, other, merged_conf, conf_b)
            merged_conf = None
        return merged

    def _doc_score(self, text: str, conf_lines: list[list[float]] | None) -> float:
        """Whole-document sort key: mean confidence if available, else heuristic."""
        if conf_lines:
            flat = [c for line in conf_lines for c in line]
            if flat:
                return float(np.mean(flat))
        return self._score_text(text)

    def _merge_two(
        self,
        a: str,
        b: str,
        conf_a: list[list[float]] | None = None,
        conf_b: list[list[float]] | None = None,
    ) -> str:
        """Merge two OCR results using line-level then word-level alignment."""
        lines_a = a.splitlines()
        lines_b = b.splitlines()

        matcher = SequenceMatcher(None, lines_a, lines_b, autojunk=False)
        out: list[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                out.extend(lines_a[i1:i2])
            elif tag == "replace":
                ca = conf_a[i1:i2] if conf_a is not None else None
                cb = conf_b[j1:j2] if conf_b is not None else None
                out.extend(self._merge_word_level(lines_a[i1:i2], lines_b[j1:j2], ca, cb))
            elif tag == "delete":
                out.extend(lines_a[i1:i2])
            elif tag == "insert":
                out.extend(lines_b[j1:j2])

        return "\n".join(out)

    def _merge_word_level(
        self,
        lines_a: list[str],
        lines_b: list[str],
        conf_a: list[list[float]] | None = None,
        conf_b: list[list[float]] | None = None,
    ) -> list[str]:
        """
        Resolve disagreements between two line blocks at word level.

        Joins each block into a word sequence, aligns with SequenceMatcher, and
        picks the better word segment for each differing region — by mean
        confidence when both segments have it, otherwise by text heuristic.
        The result is reflowed using the longer block's word-count proportions.
        """
        words_a = " ".join(lines_a).split()
        words_b = " ".join(lines_b).split()

        if not words_a:
            return lines_b
        if not words_b:
            return lines_a

        flat_a = self._flatten_conf(conf_a, len(words_a))
        flat_b = self._flatten_conf(conf_b, len(words_b))

        matcher = SequenceMatcher(None, words_a, words_b, autojunk=False)
        merged: list[str] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                merged.extend(words_a[i1:i2])
            elif tag == "replace":
                seg_a, seg_b = words_a[i1:i2], words_b[j1:j2]
                ca = flat_a[i1:i2] if flat_a is not None else None
                cb = flat_b[j1:j2] if flat_b is not None else None
                merged.extend(self._choose_segment(seg_a, seg_b, ca, cb))
            elif tag == "delete":
                merged.extend(words_a[i1:i2])
            elif tag == "insert":
                merged.extend(words_b[j1:j2])

        ref = lines_a if len(lines_a) >= len(lines_b) else lines_b
        return _reflow(merged, ref)

    @staticmethod
    def _flatten_conf(conf_lines: list[list[float]] | None, n_words: int) -> list[float] | None:
        """Flatten per-line confidences to a per-word list; None if absent or
        length-mismatched (graceful heuristic fallback)."""
        if conf_lines is None:
            return None
        flat = [c for line in conf_lines for c in line]
        return flat if len(flat) == n_words else None

    def _choose_segment(
        self,
        seg_a: list[str],
        seg_b: list[str],
        conf_a: list[float] | None,
        conf_b: list[float] | None,
    ) -> list[str]:
        """Pick the winning word segment: by mean confidence when both have it
        (text heuristic breaks ties within epsilon), else by text heuristic."""
        ca = self._mean_conf(conf_a)
        cb = self._mean_conf(conf_b)
        if ca is not None and cb is not None:
            if abs(ca - cb) < 0.05:
                return seg_a if self._score_words(seg_a) >= self._score_words(seg_b) else seg_b
            return seg_a if ca >= cb else seg_b
        return seg_a if self._score_words(seg_a) >= self._score_words(seg_b) else seg_b

    @staticmethod
    def _mean_conf(conf: list[float] | None) -> float | None:
        if not conf:
            return None
        return float(np.mean(conf))

    def _score_words(self, words: list[str]) -> float:
        """Score a word sequence: prefer high alphanumeric ratio, penalise noise tokens."""
        if not words:
            return 0.0
        text = " ".join(words)
        alnum = sum(c.isalnum() for c in text)
        # Penalty: tokens made entirely of non-alnum chars (e.g. "|", "||", "@#")
        # are classic OCR misreads and indicate a noisy segment.
        noise_tokens = sum(1 for w in words if w and not any(c.isalnum() for c in w))
        penalty = noise_tokens * 0.5
        return (alnum / len(text)) * 10 + len(text) * 0.01 - penalty

    def _score_text(self, text: str) -> float:
        """Score a full OCR result for use as a sort key (higher = better base)."""
        if not text:
            return 0.0
        words = text.split()
        alnum = sum(c.isalnum() for c in text)
        noise_tokens = sum(1 for w in words if w and not any(c.isalnum() for c in w))
        penalty = noise_tokens * 0.5
        return (alnum / len(text)) * 10 + len(text) * 0.01 - penalty

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
