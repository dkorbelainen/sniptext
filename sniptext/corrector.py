"""Text correction module for OCR post-processing using statistical spell checking."""

import re

from loguru import logger


def detect_dominant_language(text: str, candidates: list[str]) -> str:
    """
    Detect the dominant language of *text* from a list of candidate language codes.

    Uses Unicode script heuristics — no external library required.
    Works for any language combination supported by Tesseract/EasyOCR.

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

    # Score each candidate by counting how many characters in *text* belong
    # to the Unicode scripts associated with that language.
    scores = {code: 0 for code in candidates}

    for ch in letter_chars:
        cp = ord(ch)
        for code in candidates:
            if _char_matches_lang(cp, code):
                scores[code] += 1

    best = max(scores, key=lambda c: scores[c])
    # Only switch away from the first candidate if there is clear evidence
    if scores[best] > 0:
        if best != candidates[0]:
            logger.debug(
                f"Auto-detected language: {best!r} "
                f"(scores: { {k: v for k, v in scores.items()} })"
            )
        return best

    return candidates[0]


def _char_matches_lang(cp: int, lang_code: str) -> bool:
    """Return True if codepoint *cp* belongs to the script of *lang_code*."""
    # Normalise to lower-case Tesseract-style code
    code = lang_code.lower().strip()

    # ── Latin-script languages ──────────────────────────────────────────────
    _LATIN = (
        "eng", "en", "fra", "fr", "deu", "de", "spa", "es", "por", "pt",
        "ita", "it", "nld", "nl", "pol", "pl", "swe", "sv", "dan", "da",
        "nor", "nb", "fin", "fi", "hun", "hu", "ces", "cs", "slk", "sk",
        "ron", "ro", "hrv", "hr", "slv", "sl", "lit", "lt", "lav", "lv",
        "est", "et", "tur", "tr", "ind", "id", "msa", "ms", "vie", "vi",
        "afr", "af", "swa", "sw", "lat", "la",
    )
    if code in _LATIN:
        # Basic Latin + Latin-1 Supplement + Latin Extended A/B
        return (0x0041 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F)

    # ── Cyrillic-script languages ───────────────────────────────────────────
    _CYRILLIC = ("rus", "ru", "bul", "bg", "ukr", "uk", "bel", "be", "mkd", "mk", "srp", "sr")
    if code in _CYRILLIC:
        return 0x0400 <= cp <= 0x04FF

    # ── Arabic-script languages ─────────────────────────────────────────────
    _ARABIC = ("ara", "ar", "fas", "fa", "urd", "ur", "pus", "ps")
    if code in _ARABIC:
        return (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F)

    # ── CJK / Chinese ───────────────────────────────────────────────────────
    if code in ("chi_sim", "chi_tra", "zho", "zh"):
        return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)

    # ── Japanese ────────────────────────────────────────────────────────────
    if code in ("jpn", "ja"):
        return (
            (0x3040 <= cp <= 0x309F)   # Hiragana
            or (0x30A0 <= cp <= 0x30FF)  # Katakana
            or (0x4E00 <= cp <= 0x9FFF)  # Kanji (shared with Chinese)
        )

    # ── Korean ──────────────────────────────────────────────────────────────
    if code in ("kor", "ko"):
        return (0xAC00 <= cp <= 0xD7FF) or (0x1100 <= cp <= 0x11FF)

    # ── Hebrew ──────────────────────────────────────────────────────────────
    if code in ("heb", "he"):
        return 0x0590 <= cp <= 0x05FF

    # ── Greek ───────────────────────────────────────────────────────────────
    if code in ("ell", "el"):
        return 0x0370 <= cp <= 0x03FF

    # ── Thai ────────────────────────────────────────────────────────────────
    if code in ("tha", "th"):
        return 0x0E00 <= cp <= 0x0E7F

    # ── Devanagari (Hindi, Sanskrit, …) ─────────────────────────────────────
    if code in ("hin", "hi", "san", "sa", "mar", "mr", "nep", "ne"):
        return 0x0900 <= cp <= 0x097F

    # Unknown language code — never matches (fall back to first candidate)
    return False


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
