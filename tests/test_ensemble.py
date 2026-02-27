"""Tests for EnsembleOCR and post_process_text."""

import pytest

from sniptext.ensemble import EnsembleOCR, post_process_text


@pytest.fixture
def ensemble():
    return EnsembleOCR()


class TestCombineResults:
    def test_empty_list_returns_empty(self, ensemble):
        assert ensemble.combine_results([]) == ""

    def test_single_result_returned_as_is(self, ensemble):
        assert ensemble.combine_results(["hello world"]) == "hello world"

    def test_all_empty_strings_returns_empty(self, ensemble):
        assert ensemble.combine_results(["", "  ", ""]) == ""

    def test_two_identical_results(self, ensemble):
        result = ensemble.combine_results(["hello world", "hello world"])
        assert result == "hello world"

    def test_two_different_results_returns_string(self, ensemble):
        result = ensemble.combine_results(["hello world", "he1lo world"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multiline_results(self, ensemble):
        r1 = "line one\nline two"
        r2 = "line one\nline two"
        result = ensemble.combine_results([r1, r2])
        assert "line one" in result
        assert "line two" in result


class TestCalculateConfidence:
    def test_single_result_returns_half(self, ensemble):
        assert ensemble.calculate_confidence(["hello"]) == 0.5

    def test_identical_results_high_confidence(self, ensemble):
        score = ensemble.calculate_confidence(["hello world", "hello world"])
        assert score == 1.0

    def test_completely_different_results_low_confidence(self, ensemble):
        score = ensemble.calculate_confidence(["aaaa", "zzzz"])
        assert score < 0.5

    def test_empty_list_returns_half(self, ensemble):
        assert ensemble.calculate_confidence([]) == 0.5


class TestPostProcessText:
    def test_empty_returns_empty(self):
        assert post_process_text("") == ""

    def test_basic_correction_applied(self):
        result = post_process_text("1 am happy", language="eng", enable_correction=True)
        assert "I am" in result

    def test_no_correction_still_cleans_spaces(self):
        result = post_process_text("hello  world", language="eng", enable_correction=False)
        assert result == "hello world"

    def test_no_correction_fixes_punctuation(self):
        result = post_process_text("hello , world", language="eng", enable_correction=False)
        assert result == "hello, world"

    def test_no_correction_collapses_multiple_empty_lines(self):
        # Single blank line (paragraph break) must be preserved
        result = post_process_text("line one\n\nline two", language="eng", enable_correction=False)
        assert "line one" in result
        assert "line two" in result
        # Multiple consecutive blank lines should be collapsed to one
        result2 = post_process_text("a\n\n\n\nb", language="eng", enable_correction=False)
        assert "\n\n\n" not in result2

    def test_returns_string(self):
        result = post_process_text("some text", language="eng")
        assert isinstance(result, str)

    def test_numbered_list_not_penalised(self, ensemble):
        """Lines starting with a digit ('1. Item') must not be downscored."""
        results = ["1. First item\n2. Second item", "1. First item\n2. Second item"]
        result = ensemble.combine_results(results)
        assert "1." in result
        assert "2." in result

    def test_single_capital_not_penalised(self, ensemble):
        """Lines where the only word is a single capital ('I', 'A') must not be removed."""
        results = ["I went home", "I went home"]
        result = ensemble.combine_results(results)
        assert result.startswith("I")
