"""Text pool construction (no network)."""

from benchmarks.corpus import (
    build_items,
    clean_sentences,
    parse_code_lines,
    parse_conllu_sentences,
)

CONLLU = """# sent_id = 1
# text = The sheikh has been attacked near the river.
1\tThe\tthe\tDET
# sent_id = 2
# text = Too short.
# text = Барыкина иногда называют &quot;отцом&quot; русского регги.
# text = Безгачиха -- деревня в Бабушкинском районе Вологодской области.
"""


def test_parse_conllu_sentences():
    assert parse_conllu_sentences(CONLLU) == [
        "The sheikh has been attacked near the river.",
        "Too short.",
        "Барыкина иногда называют &quot;отцом&quot; русского регги.",
        "Безгачиха -- деревня в Бабушкинском районе Вологодской области.",
    ]


def test_clean_sentences_filters_by_language_length_and_charset():
    sentences = parse_conllu_sentences(CONLLU) + ["Naïve café visits are nice today."]
    assert clean_sentences(sentences, "en") == ["The sheikh has been attacked near the river."]
    assert clean_sentences(sentences, "ru") == [
        "Безгачиха -- деревня в Бабушкинском районе Вологодской области."
    ]


def test_clean_sentences_drops_duplicates():
    s = "One two three four five."
    assert clean_sentences([s, s], "en") == [s]


def test_parse_code_lines():
    raw = (
        'import os\n\n# comment\ndef f(x):\n    """Doc."""\n    return x + 1\n' + "x = " + "1" * 80
    )
    assert parse_code_lines(raw) == ["import os", "def f(x):", "    return x + 1"]


def _pool():
    en = [f"Sentence number {i} is written in plain English words." for i in range(400)]
    ru = [f"Предложение номер {i} написано простыми русскими словами." for i in range(400)]
    code = [f"value_{i} = compute(value_{i - 1}, {i})" for i in range(600)]
    return en, ru, code


def test_build_items_is_deterministic_and_unique():
    a = build_items(*_pool(), seed=1, n_prose=20, n_code=10, n_ui=5)
    b = build_items(*_pool(), seed=1, n_prose=20, n_code=10, n_ui=5)
    c = build_items(*_pool(), seed=2, n_prose=20, n_code=10, n_ui=5)
    assert a == b
    assert a != c
    assert len({item.text_id for item in a}) == len(a)


def test_build_items_counts_and_shape():
    items = build_items(*_pool(), seed=1, n_prose=20, n_code=10, n_ui=5)
    by = {}
    for item in items:
        by[(item.lang, item.content)] = by.get((item.lang, item.content), 0) + 1
        assert 1 <= len(item.text.split("\n")) <= 8
        assert item.text.strip()
    assert by == {
        ("en", "prose"): 20,
        ("ru", "prose"): 20,
        ("en", "code"): 10,
        ("en", "ui"): 5,
        ("ru", "ui"): 5,
    }


def test_no_sentence_is_shared_between_prose_items():
    items = [i for i in build_items(*_pool(), seed=1, n_prose=50, n_code=0, n_ui=0)]
    seen = set()
    for item in items:
        for number in [w for w in item.text.replace("\n", " ").split() if w.isdigit()]:
            assert (item.lang, number) not in seen
            seen.add((item.lang, number))
