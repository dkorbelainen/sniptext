"""Text postprocessing module."""

import re
from typing import List, Tuple
from loguru import logger


class TextPostprocessor:
    """Postprocess OCR results using heuristics and pattern matching."""

    def __init__(self):
        """Initialize postprocessor."""
        self.common_ocr_errors = {
            # Common Cyrillic OCR errors
            'rе': 'те',  # Latin r/e -> Cyrillic т/е
            '0': 'О',    # Zero -> O in context
            '1': 'I',    # One -> I in context
            '3': 'З',    # Three -> З in context
            '6': 'б',    # Six -> б in context
            '4': 'ч',    # Four -> ч in context
        }

    def process(self, text: str, language: str = "multi") -> str:
        """
        Apply postprocessing to OCR text.

        Args:
            text: Raw OCR output
            language: Language code

        Returns:
            Cleaned text
        """
        if not text:
            return text

        original_length = len(text)

        # Apply corrections
        text = self._fix_mixed_scripts(text)
        text = self._fix_common_errors(text)
        text = self._fix_spacing(text)
        text = self._fix_punctuation(text)

        corrections = original_length - len(text)
        if corrections:
            logger.debug(f"Applied {corrections} postprocessing corrections")

        return text

    def _fix_mixed_scripts(self, text: str) -> str:
        """Fix mixed Latin/Cyrillic characters in words."""
        # Common Latin chars that should be Cyrillic
        replacements = {
            'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р',
            'c': 'с', 'y': 'у', 'x': 'х', 'B': 'В',
            'H': 'Н', 'K': 'К', 'M': 'М', 'T': 'Т',
        }

        # Only replace if surrounded by Cyrillic
        for latin, cyrillic in replacements.items():
            # Pattern: Cyrillic + latin + Cyrillic
            pattern = r'([а-яА-ЯёЁ])' + re.escape(latin) + r'([а-яА-ЯёЁ])'
            text = re.sub(pattern, r'\1' + cyrillic + r'\2', text)

        return text

    def _fix_common_errors(self, text: str) -> str:
        """Fix common OCR recognition errors."""
        # Fix spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Fix double spaces
        text = re.sub(r' {2,}', ' ', text)

        # Fix line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues."""
        # Remove spaces at line start/end
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        return '\n'.join(lines)

    def _fix_punctuation(self, text: str) -> str:
        """Fix punctuation issues."""
        # Ensure space after punctuation
        text = re.sub(r'([.,!?])([а-яА-ЯёЁa-zA-Z])', r'\1 \2', text)

        # Fix quotes
        text = re.sub(r',,', '"', text)
        text = re.sub(r"''", '"', text)

        return text


class ContextAwareCorrector:
    """Context-aware text correction using statistical methods."""

    def __init__(self):
        """Initialize corrector."""
        self.word_freq = {}
        self.bigrams = {}

    def correct(self, text: str) -> str:
        """
        Apply context-aware corrections.

        Args:
            text: Input text

        Returns:
            Corrected text
        """
        # For now, simple implementation
        # Can be extended with language models
        return text

    def train_on_text(self, text: str) -> None:
        """Train corrector on sample text."""
        words = re.findall(r'\b\w+\b', text.lower())

        # Build word frequency
        for word in words:
            self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Build bigrams
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1
