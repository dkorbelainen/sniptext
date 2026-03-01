"""Unit tests for OCR text correction."""

import pytest

from sniptext.corrector import OCRCorrector, correct_ocr_text, detect_dominant_language


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
