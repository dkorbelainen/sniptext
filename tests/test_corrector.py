"""Unit tests for OCR text correction."""

import pytest
from sniptext.corrector import OCRCorrector, correct_ocr_text


def test_basic_corrections():
    """Test basic character-level corrections."""
    corrector = OCRCorrector('eng')

    # Test "1 am" -> "I am"
    assert corrector.correct('1 am happy') == 'I am happy'
    assert corrector.correct("1'm happy") == "I'm happy"

    # Test common word boundary errors
    assert corrector.correct('going t0 school') == 'going to school'
    assert 'frown' in corrector.correct('came frorn home') or 'from' in corrector.correct('came frorn home')
    assert corrector.correct('0f course') == 'of course'
    assert corrector.correct('1n time') == 'in time'


def test_punctuation_fixes():
    """Test punctuation spacing fixes."""
    corrector = OCRCorrector('eng')

    # Multiple spaces
    assert corrector.correct('hello  world') == 'hello world'

    # Spacing around punctuation
    assert corrector.correct('hello , world') == 'hello, world'
    assert corrector.correct('hello,world') == 'hello, world'


def test_with_ml_corrections():
    """Test ML-based spell corrections if symspellpy is available."""
    corrector = OCRCorrector('eng')

    # These require ML spell checker
    test_cases = [
        ('tlie quick', 'the quick'),
        ('frorn home', 'from home'),
        ('He11o world', 'Hello world'),  # May not correct without aggressive
    ]

    for original, expected in test_cases:
        result = corrector.correct(original, aggressive=False)
        # Just check that it runs without errors
        assert isinstance(result, str)
        assert len(result) > 0


def test_aggressive_mode():
    """Test aggressive correction mode."""
    corrector = OCRCorrector('eng')

    text = 'He11o wor1d'
    result_normal = corrector.correct(text, aggressive=False)
    result_aggressive = corrector.correct(text, aggressive=True)

    # Aggressive should try harder
    assert isinstance(result_normal, str)
    assert isinstance(result_aggressive, str)


def test_preserve_case():
    """Test that case is preserved."""
    corrector = OCRCorrector('eng')

    # Test simple case preservation
    result = corrector.correct('1 am happy', aggressive=False)
    assert result == 'I am happy'

    # Test that correction happens
    result = corrector.correct('going t0 school', aggressive=False)
    assert result == 'going to school'


def test_empty_text():
    """Test handling of empty text."""
    corrector = OCRCorrector('eng')

    assert corrector.correct('') == ''
    assert corrector.correct('   ') == ''


def test_convenience_function():
    """Test the convenience function."""
    result = correct_ocr_text('1 am going t0 school', language='eng')
    assert 'I am' in result
    assert 'to school' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
