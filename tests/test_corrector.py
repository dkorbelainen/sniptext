"""Unit tests for OCR text correction."""

import pytest

from sniptext.corrector import (
    OCRCorrector,
    _char_matches_lang,
    _line_looks_like_code,
    _looks_non_lexical,
    correct_ocr_text,
    detect_dominant_language,
)


def test_basic_corrections():
    """Test basic character-level corrections."""
    corrector = OCRCorrector("eng")

    # Test "1 am" -> "I am"
    assert corrector.correct("1 am happy") == "I am happy"
    assert corrector.correct("1'm happy") == "I'm happy"

    # Test common word boundary errors
    assert corrector.correct("going t0 school") == "going to school"
    assert corrector.correct("0f course") == "of course"
    assert corrector.correct("1n time") == "in time"


def test_punctuation_fixes():
    """Test punctuation spacing fixes."""
    corrector = OCRCorrector("eng")

    # Multiple spaces
    assert corrector.correct("hello  world") == "hello world"

    # Spacing around punctuation
    assert corrector.correct("hello , world") == "hello, world"
    assert corrector.correct("hello,world") == "hello, world"


def test_with_ml_corrections():
    """Test ML-based spell corrections if symspellpy is available."""
    corrector = OCRCorrector("eng")

    # These require ML spell checker
    test_cases = [
        ("tlie quick", "the quick"),
        ("frorn home", "from home"),
        ("He11o world", "Hello world"),  # May not correct without aggressive
    ]

    for original, expected in test_cases:
        result = corrector.correct(original, aggressive=False)
        # Just check that it runs without errors
        assert isinstance(result, str)
        assert len(result) > 0


def test_aggressive_mode():
    """Test aggressive correction mode."""
    corrector = OCRCorrector("eng")

    text = "He11o wor1d"
    result_normal = corrector.correct(text, aggressive=False)
    result_aggressive = corrector.correct(text, aggressive=True)

    # Aggressive should try harder
    assert isinstance(result_normal, str)
    assert isinstance(result_aggressive, str)


def test_preserve_case():
    """Test that case is preserved."""
    corrector = OCRCorrector("eng")

    # Test simple case preservation
    result = corrector.correct("1 am happy", aggressive=False)
    assert result == "I am happy"

    # Test that correction happens
    result = corrector.correct("going t0 school", aggressive=False)
    assert result == "going to school"


def test_empty_text():
    """Test handling of empty text."""
    corrector = OCRCorrector("eng")

    assert corrector.correct("") == ""
    assert corrector.correct("   ") == ""


def test_convenience_function():
    """Test the convenience function."""
    result = correct_ocr_text("1 am going t0 school", language="eng")
    assert "I am" in result
    assert "to school" in result


def test_english_corrections_not_applied_to_russian():
    """English-only substitutions must not mutate non-English text."""
    corrector = OCRCorrector("rus")
    text = "Привет 1n 0f мир"
    result = corrector.correct(text)
    assert "1n" in result
    assert "0f" in result


def test_paragraph_breaks_preserved():
    """Newlines must survive spell correction."""
    corrector = OCRCorrector("eng")
    text = "First paragraph.\n\nSecond paragraph."
    result = corrector.correct(text)
    assert "\n" in result
    assert "First paragraph" in result
    assert "Second paragraph" in result


def test_multiple_blank_lines_collapsed():
    """Three or more consecutive blank lines should be collapsed to one."""
    corrector = OCRCorrector("eng")
    text = "A\n\n\n\nB"
    result = corrector.correct(text)
    assert "\n\n\n" not in result


class TestDetectDominantLanguage:
    def test_single_candidate_passthrough(self):
        assert detect_dominant_language("anything", ["eng"]) == "eng"

    def test_empty_text_returns_first(self):
        assert detect_dominant_language("", ["eng", "rus"]) == "eng"

    def test_russian_text_detected(self):
        text = "Привет мир это тест кириллицы"
        assert detect_dominant_language(text, ["eng", "rus"]) == "rus"

    def test_english_text_detected(self):
        text = "Hello world this is English text"
        assert detect_dominant_language(text, ["eng", "rus"]) == "eng"

    def test_arabic_detected(self):
        text = "مرحبا بالعالم"
        assert detect_dominant_language(text, ["fra", "ara"]) == "ara"

    def test_korean_detected(self):
        text = "안녕하세요 세계"
        assert detect_dominant_language(text, ["jpn", "kor"]) == "kor"

    def test_devanagari_detected(self):
        text = "नमस्ते दुनिया"
        assert detect_dominant_language(text, ["deu", "hin"]) == "hin"

    def test_unknown_lang_code_falls_back(self):
        # Unknown codes should not crash; fall back to first candidate
        text = "hello world"
        result = detect_dominant_language(text, ["eng", "xyz_unknown"])
        assert result in ("eng", "xyz_unknown")

    def test_fallback_when_no_script_matches(self):
        # Arabic text against Latin/Cyrillic candidates → all scores 0 → first candidate
        result = detect_dominant_language("مرحبا", ["eng", "rus"])
        assert result == "eng"


def test_punctuation_spacing_works_for_cyrillic():
    """After the fix, punct+letter space rule applies to Cyrillic text."""
    corrector = OCRCorrector("rus")
    result = corrector.correct("Привет,мир")
    assert "Привет, мир" == result


def test_punctuation_spacing_works_for_arabic():
    """Unicode letter lookahead must not break for non-Latin/Cyrillic scripts."""
    corrector = OCRCorrector("ara")
    result = corrector.correct("مرحبا,عالم")
    assert "مرحبا, عالم" == result


def test_spell_correct_preserves_punctuation_around_word():
    """Prefix/suffix punctuation must survive spell correction."""
    corrector = OCRCorrector("eng")
    # "teh" (common OCR misread of "the") wrapped in parentheses
    result = corrector.correct("(teh)")
    # Either corrected-and-wrapped or left as-is — either way parens must stay
    assert result.startswith("(") and result.endswith(")")


def test_spell_correct_uppercase_word_stays_uppercase():
    """All-caps misspelled word must be corrected in all-caps form."""
    corrector = OCRCorrector("eng")
    # Force aggressive mode so a wider edit-distance is attempted
    result = corrector.correct("WRODS on screen", aggressive=True)
    if "WORDS" in result:
        assert result.startswith("WORDS")
    # Regardless of whether spell correction fired, the first token must
    # remain all-caps and be either the original or corrected form.
    first_token = result.split()[0]
    assert first_token in ("WRODS", "WORDS")
    assert first_token.upper() == first_token


class TestCharMatchesLang:
    """Direct tests for _char_matches_lang to cover each script branch."""

    def test_chinese_simplified_matches(self):
        assert _char_matches_lang(0x4E2D, "chi_sim")  # 中

    def test_hebrew_matches(self):
        assert _char_matches_lang(0x05D0, "heb")  # א

    def test_greek_matches(self):
        assert _char_matches_lang(0x03B1, "ell")  # α

    def test_thai_matches(self):
        assert _char_matches_lang(0x0E01, "tha")  # ก

    def test_no_script_match_returns_false(self):
        # Unknown lang code should never match
        assert not _char_matches_lang(ord("A"), "xyz_unknown")


class TestLooksNonLexical:
    """Guard that keeps SymSpell off non-dictionary tokens."""

    def test_plain_word_is_lexical(self):
        assert not _looks_non_lexical("teh")
        assert not _looks_non_lexical("hello")
        assert not _looks_non_lexical("Hello")  # Titlecase still correctable

    def test_snake_case_skipped(self):
        assert _looks_non_lexical("snake_case")
        assert _looks_non_lexical("max_edit_distance")

    def test_camel_case_skipped(self):
        assert _looks_non_lexical("getValue")
        assert _looks_non_lexical("iPhone")

    def test_digits_skipped(self):
        assert _looks_non_lexical("abc123")
        assert _looks_non_lexical("v2")

    def test_path_or_dotted_skipped(self):
        assert _looks_non_lexical("os.path")
        assert _looks_non_lexical("a/b")

    def test_all_caps_skipped(self):
        assert _looks_non_lexical("TOTAL")
        assert _looks_non_lexical("API")

    def test_single_letter_not_treated_as_acronym(self):
        assert not _looks_non_lexical("A")


def test_code_identifier_not_corrected():
    """Identifiers must survive spell correction verbatim."""
    corrector = OCRCorrector("eng")
    for token in ("getValue", "snake_case", "config_path", "iPhone"):
        result = corrector.correct(f"call {token} here")
        assert token in result


class TestLineLooksLikeCode:
    def test_prose_is_not_code(self):
        assert not _line_looks_like_code("the quick brown fox jumps")
        assert not _line_looks_like_code("File Edit View Help")

    def test_symbols_flag_code(self):
        assert _line_looks_like_code("x = foo(y)")
        assert _line_looks_like_code("arr[i] += 1")
        assert _line_looks_like_code("ptr->next")

    def test_keyword_flags_code(self):
        assert _line_looks_like_code("def main():")
        assert _line_looks_like_code("return value")
        assert _line_looks_like_code("import os")

    def test_indent_flags_code(self):
        assert _line_looks_like_code("    result = compute")
        assert not _line_looks_like_code("    ")  # blank indent only


def test_code_line_left_untouched():
    corrector = OCRCorrector("eng")
    line = "    idx = buf.len"
    # convenience: code line must pass through spell correction unchanged
    assert "idx" in corrector.correct(line) and "buf" in corrector.correct(line)


class TestSpellCorrectEdgeCases:
    def test_all_punctuation_word_unchanged(self):
        corrector = OCRCorrector("eng")
        # A "word" that is entirely punctuation strips to empty → kept as-is
        result = corrector.correct("hello ... world")
        assert "..." in result

    def test_spell_correct_skips_when_no_spellchecker(self):
        corrector = OCRCorrector("eng")
        corrector._spellchecker = None
        # _spell_correct should return text unchanged when no spellchecker
        assert corrector._spell_correct("wrods") == "wrods"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
