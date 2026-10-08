import numpy as np
import pytest

from benchmarks import corrector_eval
from sniptext.pipelines import Pipeline


class OneAction:
    policy = "pre_ocr"
    actions = (Pipeline("a", ("light",), "6"),)

    def choose_vector(self, x):
        return 0


def data():
    return {
        "rows": [{"text": {"a": "helo"}, "gt": "hello"}, {"text": {"a": "world"}, "gt": "world"}],
        "names": ["a"],
        "features": np.zeros((2, 1)),
        "conf_stats": np.zeros((2, 1, 5)),
        "empty": np.zeros((2, 1), dtype=bool),
        "routable": np.ones(2, dtype=bool),
        "text_id": np.array(["t1", "t2"]),
    }


def test_measure_compares_corrected_with_raw_text(monkeypatch):
    monkeypatch.setattr(
        corrector_eval, "post_process_text", lambda text, **kwargs: text.replace("helo", "hello")
    )
    out = corrector_eval.measure(data(), OneAction(), np.array([True, True]), "eng")
    assert out["n"] == 2 and out["changed_share"] == 0.5
    assert out["cer_raw"][0] == pytest.approx(0.1) and out["cer_corrected"][0] == 0.0
    assert out["delta"][0] == pytest.approx(-0.1)


def test_measure_passes_the_language_and_the_safe_mode(monkeypatch):
    seen = []

    def fake(text, language, enable_correction, aggressive):
        seen.append((language, enable_correction, aggressive))
        return text

    monkeypatch.setattr(corrector_eval, "post_process_text", fake)
    corrector_eval.measure(data(), OneAction(), np.array([True, False]), "eng+rus")
    assert seen == [("eng+rus", True, False)]


def test_decision_needs_an_interval_below_zero():
    assert corrector_eval.decide((-0.01, -0.02, -0.001)) == "keep"
    assert corrector_eval.decide((-0.01, -0.02, 0.0)) == "remove"
    assert corrector_eval.decide((0.01, 0.001, 0.02)) == "remove"
