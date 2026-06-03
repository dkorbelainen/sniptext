import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import aggregate_cer, cer, levenshtein, normalize_text, wer


def test_normalize_collapses_whitespace_keeps_case():
    assert normalize_text("a  b\n  c\t d ") == "a b c d"
    assert normalize_text("Hello   World") == "Hello World"


def test_normalize_empty():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t ") == ""


def test_levenshtein_identical():
    assert levenshtein("abc", "abc") == 0


def test_levenshtein_single_substitution():
    assert levenshtein("abc", "abd") == 1


def test_levenshtein_insert_delete():
    assert levenshtein("abc", "ab") == 1
    assert levenshtein("ab", "abc") == 1


def test_cer_identical_is_zero():
    assert cer("hello world", "hello world") == 0.0


def test_cer_single_char_error():
    assert cer("helxo", "hello") == 1 / 5


def test_cer_empty_gt_empty_pred():
    assert cer("", "") == 0.0


def test_cer_empty_gt_nonempty_pred():
    assert cer("abc", "") == 1.0


def test_wer_identical_is_zero():
    assert wer("a b c", "a b c") == 0.0


def test_wer_one_word_wrong():
    assert wer("a x c", "a b c") == 1 / 3


def test_aggregate_cer_averages_per_sample():
    pairs = [("hello", "hello"), ("ab", "abcd")]
    assert aggregate_cer(pairs) == 0.25
