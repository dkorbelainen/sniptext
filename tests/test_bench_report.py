"""Report rendering from a fabricated evaluation (no OCR, no plotting)."""

import re

import pytest
from test_bench_train import FAST, fake_run

from benchmarks import report, train_router
from sniptext.pipelines import V04

pytestmark = pytest.mark.usefixtures("fast_bootstrap")

MEASURE = {"n": 10, "cer_raw": [0.050, 0.040, 0.060], "cer_corrected": [0.055, 0.045, 0.065],
           "delta": [0.005, -0.001, 0.011], "changed_share": 0.2}  # fmt: skip
CORRECTOR = {"language": "eng", "symspell": True, "val": MEASURE, "test": MEASURE,
             "decision": "remove"}  # fmt: skip
ENVIRONMENT = {"chrome": "Google Chrome 1.0", "fonts": {"sans-serif": "A.ttf", "serif": "B.ttf",
                                                         "monospace": "C.ttf"}}  # fmt: skip
HEADINGS = (
    "## Task", "## Data", "## Result on held-out texts", "## Unseen fonts",
    "## Pages rendered by a browser", "## Confirmation on fresh texts",
    "## Images with and without added noise", "## Clean and degraded images",
    "## Accuracy against time",
    "## How the pipelines were chosen", "## Model selection", "## Features", "## Breakdown",
    "## Why EasyOCR was removed", "## Text correction", "## Limitations", "## Reproduce",
)  # fmt: skip


def evaluation(tmp_path, **kwargs):
    p = fake_run(tmp_path, **kwargs)
    return train_router.run(p["results"], p["timing"], p["legacy"], p["model"], p["eval"],
                            candidates=FAST)  # fmt: skip


def test_render_has_every_section_and_no_missing_numbers(tmp_path):
    text = report.render(evaluation(tmp_path), CORRECTOR, ENVIRONMENT, "abc1234", "2026-10-07")
    for heading in HEADINGS:
        assert heading in text, heading
    assert "## Out-of-domain receipts" not in text  # the fabricated run has no such slice
    assert not re.search(r"\bnan\b|None|\{|\}", text)
    assert "abc1234" in text and "(shipped)" in text and "Google Chrome 1.0" in text
    assert "`p_base`" in text and "`p_alt`" in text


def test_criteria_are_stated_with_their_numbers(tmp_path):
    text = report.render(evaluation(tmp_path), CORRECTOR, ENVIRONMENT, "c", "d")
    assert "1. The shipped router has significantly lower CER" in text
    assert text.count("**met**") + text.count("**not met**") == 4
    assert "**met** (paired difference -" in text
    assert "4. On fresh texts" in text


def test_a_static_model_is_reported_as_such(tmp_path):
    text = report.render(evaluation(tmp_path, helpful=False), CORRECTOR, None, "c", "d")
    assert "No router ships" in text
    for heading in ("## Accuracy against time", "## Model selection", "## Features"):
        assert heading not in text
    assert "**not met**" in text and "Fonts:" not in text


def test_the_corrector_section_follows_the_decision(tmp_path):
    ev = evaluation(tmp_path)
    removed = report.render(ev, CORRECTOR, None, "c", "d")
    assert "was removed" in removed
    assert "`benchmarks/corrector_eval.json`, measured before" in removed
    kept = {**CORRECTOR, "decision": "keep"}
    assert "stays on by default" in report.render(ev, kept, None, "c", "d")


def summary(delta):
    return {"policy": "router_pre_ocr", "cer": (0.05, 0.04, 0.06), "delta": delta}


def test_verdict_states_what_the_interval_supports():
    ev = {"best_static": "p_base", "shipped": "pre_ocr"}
    assert "lower than always running `p_base` by 0.020" in report.verdict(
        summary((-0.02, -0.03, -0.01)), ev
    )
    assert "higher than always running `p_base` by 0.020" in report.verdict(
        summary((0.02, 0.01, 0.03)), ev
    )
    assert "not distinguishable from always running `p_base`" in report.verdict(
        summary((-0.01, -0.03, 0.01)), ev
    )


def test_the_difference_to_the_v04_pipeline_is_stated_when_it_was_measured(tmp_path):
    p = fake_run(tmp_path)
    for key in ("results", "timing"):
        p[key].write_text(p[key].read_text().replace("p_bad", V04.name))
    ev = train_router.run(p["results"], p["timing"], p["legacy"], p["model"], p["eval"],
                          candidates=FAST)  # fmt: skip
    text = report.render(ev, CORRECTOR, None, "c", "d")
    assert text.count("Against Tesseract as version 0.4 ran it") == len(ev["slices"])
    assert "Against Tesseract as version 0.4" not in report.render(
        evaluation(tmp_path / "plain"), CORRECTOR, None, "c", "d"
    )


def test_small_bounds_are_not_printed_as_zero():
    assert report._signed((-0.0117, -0.02403, 0.00004)) == "-0.012 [-0.024, +0.00004]"
    assert report._signed((-0.0117, -0.02403, -0.00001)) == "-0.012 [-0.024, -0.00001]"
    assert report._signed((0.0, 0.0, 0.0)) == "+0.000 [+0.000, +0.000]"
    assert report._ci((0.0669, 0.0541, 0.0812)) == "0.067 [0.054, 0.081]"


def test_the_change_of_the_selection_rule_is_disclosed(tmp_path):
    ev = evaluation(tmp_path)
    ev["criteria"]["preregistered_policy"] = "pre_ocr"
    ev["shipped"] = "cascade"
    text = report.render(ev, CORRECTOR, None, "c", "d")
    assert "The rule fixed before the measurement" in text
    assert "after the held-out results were seen" in text
    same = evaluation(tmp_path / "same")
    same["criteria"]["preregistered_policy"] = same["shipped"]
    assert "after the held-out results were seen" not in report.render(
        same, CORRECTOR, None, "c", "d"
    )


def test_the_noise_split_shows_both_routers_with_intervals(tmp_path):
    text = report.render(evaluation(tmp_path), CORRECTOR, None, "c", "d")
    section = text.split("## Images with and without added noise")[1].split("\n## ")[0]
    assert "with added noise" in section and "without added noise" in section
    assert "Router, before OCR" in section and "Router, cascade" in section
    assert section.count("[") >= 4


def test_limitations_state_the_fitted_size_range_and_the_disclosures(tmp_path):
    text = report.render(evaluation(tmp_path), CORRECTOR, None, "c", "d")
    limitations = text.split("## Limitations")[1]
    assert "0.001 megapixels" in limitations
    assert "seen on the held-out slice of the 0.4 run" in limitations
    assert "after the timings of the receipts above the limit were seen" in limitations
