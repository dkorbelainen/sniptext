"""Text correction module for OCR post-processing using statistical spell checking."""

import re

from loguru import logger


def detect_dominant_language(text: str, candidates: list[str]) -> str:
    """
    Detect the dominant language of *text* from a list of candidate language codes.

    Uses character-set heuristics — no external library required.
    Supported candidate codes: any subset of {"eng","en","rus","ru"}.
    Falls back to the first candidate when detection is inconclusive.

    Args:
        text: OCR text to analyse.
        candidates: Language codes to choose from (Tesseract or EasyOCR format).

    Returns:
        The best-matching language code from *candidates*.
    """
    if len(candidates) == 1:
        return candidates[0]

    letter_chars = [c for c in text if c.isalpha()]
    if not letter_chars:
        return candidates[0]

    cyrillic = sum(1 for c in letter_chars if "\u0400" <= c <= "\u04ff")
    latin = sum(1 for c in letter_chars if c.isascii())
    total = len(letter_chars)

    cyrillic_ratio = cyrillic / total
    latin_ratio = latin / total

    # Map candidate codes to script families
    ru_codes = {"rus", "ru"}
    en_codes = {"eng", "en"}

    has_ru = bool(set(candidates) & ru_codes)
    has_en = bool(set(candidates) & en_codes)

    if has_ru and has_en:
        dominant = "rus" if cyrillic_ratio >= latin_ratio else "eng"
        for code in candidates:
            if dominant in ("rus", "ru") and code in ru_codes:
                logger.debug(
                    f"Auto-detected language: Russian "
                    f"(cyrillic={cyrillic_ratio:.0%}, latin={latin_ratio:.0%})"
                )
                return code
            if dominant in ("eng", "en") and code in en_codes:
                logger.debug(
                    f"Auto-detected language: English "
                    f"(cyrillic={cyrillic_ratio:.0%}, latin={latin_ratio:.0%})"
                )
                return code

    return candidates[0]


class OCRCorrector:
    """Corrects OCR errors using dictionary-based spell checking."""

    def __init__(self, language: str = "eng"):
        """
        Initialize corrector.

        Args:
            language: Language code (eng, rus, etc.)
        """
        self.language = language
        self._spellchecker = None
        self._initialized = False

    def _lazy_init(self):
        """Initialize spellchecker on first use."""
        if self._initialized:
            return

        try:
            from symspellpy import SymSpell

            self._spellchecker = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

            if self.language in ["eng", "en"]:
                # Use importlib.resources instead of deprecated pkg_resources
                try:
                    from importlib.resources import files

                    dict_path = str(
                        files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
                    )
                except (ImportError, AttributeError):
                    # Fallback for older Python versions
                    import pkg_resources

                    dict_path = pkg_resources.resource_filename(
                        "symspellpy", "frequency_dictionary_en_82_765.txt"
                    )

                self._spellchecker.load_dictionary(dict_path, term_index=0, count_index=1)
                logger.debug("Loaded English dictionary for spell correction")
            else:
                logger.debug(f"Spell correction not available for language: {self.language}")
                self._spellchecker = None

            self._initialized = True

        except ImportError:
            logger.debug("symspellpy not installed, using basic correction only")
            self._spellchecker = None
            self._initialized = True
        except Exception as e:
            logger.debug(f"Could not initialize spellchecker: {e}")
            self._spellchecker = None
            self._initialized = True

    def correct(self, text: str, aggressive: bool = False) -> str:
        """
        Apply correction to OCR text.

        Args:
            text: Raw OCR text
            aggressive: Apply more aggressive corrections

        Returns:
            Corrected text
        """
        if not text:
            return text

        self._lazy_init()

        original_text = text

        text = self._fix_obvious_errors(text)

        if self._spellchecker:
            text = self._spell_correct(text, aggressive=aggressive)

        text = self._final_cleanup(text)

        if text != original_text:
            logger.debug("Applied text corrections")

        return text

    def _fix_obvious_errors(self, text: str) -> str:
        """Fix obvious character-level OCR errors."""
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\s+([.,!?;:)])", r"\1", text)
        text = re.sub(r"([.,!?;:])(?=[a-zA-Zа-яА-ЯёЁ])", r"\1 ", text)

        # English-only single-character and word-boundary corrections.
        # These patterns use English words so they must not run on non-English text.
        if self.language in ("eng", "en"):
            text = re.sub(r"\b1\s+am\b", "I am", text, flags=re.IGNORECASE)
            text = re.sub(r"\b1\s+have\b", "I have", text, flags=re.IGNORECASE)
            text = re.sub(r"\b1\s+will\b", "I will", text, flags=re.IGNORECASE)
            text = re.sub(r"\b1\'m\b", "I'm", text)
            text = re.sub(r"\b1\'ve\b", "I've", text)
            text = re.sub(r"\b1\'ll\b", "I'll", text)

            text = re.sub(r"\b0f\b", "of", text)
            text = re.sub(r"\b0r\b", "or", text)
            text = re.sub(r"\b1n\b", "in", text)
            text = re.sub(r"\bt0\b", "to", text)

        return text

    def _spell_correct(self, text: str, aggressive: bool = False) -> str:
        """Apply dictionary-based spell correction."""
        if not self._spellchecker:
            return text

        try:
            from symspellpy import Verbosity
        except ImportError:
            return text

        # Process line by line to preserve paragraph structure.
        # text.split() would collapse newlines into spaces.
        corrected_lines = []
        for line in text.split("\n"):
            corrected_lines.append(self._spell_correct_line(line, aggressive, Verbosity))
        return "\n".join(corrected_lines)

    def _spell_correct_line(self, line: str, aggressive: bool, Verbosity) -> str:
        """Spell-correct a single line of text."""
        words = line.split(" ")
        corrected_words = []

        for word in words:
            if not word or len(word) < 3 or not any(c.isalpha() for c in word):
                corrected_words.append(word)
                continue

            prefix = ""
            suffix = ""
            clean_word = word

            while clean_word and not clean_word[0].isalnum():
                prefix += clean_word[0]
                clean_word = clean_word[1:]

            while clean_word and not clean_word[-1].isalnum():
                suffix = clean_word[-1] + suffix
                clean_word = clean_word[:-1]

            if not clean_word:
                corrected_words.append(word)
                continue

            max_edit_distance = 2 if aggressive else 1
            suggestions = self._spellchecker.lookup(
                clean_word.lower(), Verbosity.CLOSEST, max_edit_distance=max_edit_distance
            )

            if suggestions and len(suggestions) > 0:
                suggestion = suggestions[0]

                if suggestion.distance > 0 and suggestion.count > 10:
                    corrected = suggestion.term

                    if clean_word.isupper():
                        corrected = corrected.upper()
                    elif clean_word[0].isupper():
                        corrected = corrected.capitalize()

                    corrected_words.append(prefix + corrected + suffix)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        # Remove multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Strip trailing/leading whitespace from each line, but preserve
        # intentional blank lines (paragraph breaks) — collapse runs of
        # more than one consecutive blank line down to a single blank line.
        lines = [line.strip() for line in text.split("\n")]
        cleaned: list[str] = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank and cleaned:
                    # Keep one blank line as paragraph separator
                    cleaned.append("")
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False

        return "\n".join(cleaned).strip()

# Module-level cache so repeated calls with the same language reuse
# the already-initialised corrector (avoids reloading the SymSpell dict).
_corrector_cache: dict[str, "OCRCorrector"] = {}


def correct_ocr_text(text: str, language: str = "eng", aggressive: bool = False) -> str:
    """
    Correct OCR text errors.

    Args:
        text: Raw OCR text
        language: Language code
        aggressive: Apply aggressive corrections

    Returns:
        Corrected text
    """
    if language not in _corrector_cache:
        _corrector_cache[language] = OCRCorrector(language)
    return _corrector_cache[language].correct(text, aggressive=aggressive)
