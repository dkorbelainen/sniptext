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
    "## Method", "## Data", "## Shipped policy", "## Results", "### Fresh texts",
    "### Held-out texts", "### Unseen fonts", "### Browser pages", "## Added noise",
    "## By degradation", "## Time", "## Pipeline selection", "## Model selection",
    "## Removed components", "## Reproduce",
)  # fmt: skip


def evaluation(tmp_path, **kwargs):
    p = fake_run(tmp_path, **kwargs)
    return train_router.run(p["results"], p["timing"], p["legacy"], p["model"], p["eval"],
                            candidates=FAST)  # fmt: skip


def test_render_has_every_section_and_no_missing_numbers(tmp_path):
    text = report.render(evaluation(tmp_path), CORRECTOR, ENVIRONMENT, "abc1234", "2026-10-07")
    for heading in HEADINGS:
        assert heading in text, heading
    assert "### Receipts" not in text  # the fabricated run has no such slice
    assert not re.search(r"\bnan\b|None|\{|\}", text)
    assert "abc1234" in text and "(shipped)" in text and "Google Chrome 1.0" in text
    assert "`p_base`" in text and "`p_alt`" in text


def test_criteria_are_stated_with_their_numbers(tmp_path):
    text = report.render(evaluation(tmp_path), CORRECTOR, ENVIRONMENT, "c", "d")
    assert "1. Lower CER on held-out texts than the best static pipeline" in text
    assert text.count("**met**") + text.count("**not met**") == 4
    assert "**met** (-" in text
    assert "4. Lower CER on fresh texts" in text


def test_a_criterion_without_its_slice_is_not_judged(tmp_path):
    ev = evaluation(tmp_path)
    ev["criteria"].update(browser_not_worse=None, browser_delta=None)
    text = report.render(ev, CORRECTOR, ENVIRONMENT, "c", "d")
    assert "on browser pages: **not measured**." in text


def test_slice_tables_keep_one_row_per_policy(tmp_path):
    ev = evaluation(tmp_path)
    text = report.render(ev, CORRECTOR, ENVIRONMENT, "c", "d")
    table = text.split("### Held-out texts")[1].split("\n\n")[1].splitlines()
    assert table[0] == "| Policy | CER | Difference to best static | Time, ms |"
    assert len(table) == 2 + len(ev["slices"]["test"])


def test_model_selection_shows_the_best_candidate_of_each_kind(tmp_path):
    ev = evaluation(tmp_path)
    section = report.render(ev, CORRECTOR, None, "c", "d").split("## Model selection")[1]
    table = section.split("\n\n")[2].splitlines()
    kinds = {entry["spec"]["kind"] for entry in ev["policies"]["cascade"]["selection"]}
    assert len(table) == 2 + 2 * len(kinds)
    assert "The shipped router has" in section and "Shuffling one" in section


def test_a_static_model_is_reported_as_such(tmp_path):
    text = report.render(evaluation(tmp_path, helpful=False), CORRECTOR, None, "c", "d")
    assert "No router ships: `p_base` runs on every image." in text
    for absent in ("![CER against time]", "## Model selection", "Browser: "):
        assert absent not in text
    assert "**not met**" in text


def test_the_corrector_section_follows_the_decision(tmp_path):
    ev = evaluation(tmp_path)
    removed = report.render(ev, CORRECTOR, None, "c", "d")
    assert "was removed" in removed
    assert "`benchmarks/corrector_eval.json`, measured before" in removed
    kept = {**CORRECTOR, "decision": "keep"}
    assert "stays on by default" in report.render(ev, kept, None, "c", "d")


def test_a_relation_says_only_what_the_interval_supports():
    assert report._relation((-0.02, -0.03, -0.01)) == "lower than"
    assert report._relation((0.02, 0.01, 0.03)) == "higher than"
    assert report._relation((-0.01, -0.03, 0.01)) == "not distinguishable from"


def test_the_difference_to_the_v04_pipeline_is_stated_when_it_was_measured(tmp_path):
    p = fake_run(tmp_path)
    for key in ("results", "timing"):
        p[key].write_text(p[key].read_text().replace("p_bad", V04.name))
    ev = train_router.run(p["results"], p["timing"], p["legacy"], p["model"], p["eval"],
                          candidates=FAST)  # fmt: skip
    text = report.render(ev, CORRECTOR, None, "c", "d")
    assert text.count("Shipped policy against 0.4 Tesseract") == len(ev["slices"])
    assert "against 0.4 Tesseract" not in report.render(
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
    section = text.split("## Added noise")[1].split("\n## ")[0]
    assert "with added noise" in section and "without added noise" in section
    assert "Router, before OCR" in section and "Router, cascade" in section
    assert section.count("[") >= 4


def test_the_report_has_no_limitations_section(tmp_path):
    text = report.render(evaluation(tmp_path), CORRECTOR, None, "c", "d")
    assert "## Limitations" not in text
    assert len(text.split()) < 1900


def test_the_time_figure_names_the_router_that_ships(tmp_path):
    ev = evaluation(tmp_path)
    text = report.render(ev, CORRECTOR, None, "c", "d")
    section = text.split("## Time")[1].split("\n## ")[0]
    label = {"cascade": "Router, cascade", "pre_ocr": "Router, before OCR"}[ev["shipped"]]
    assert f"Only {label} ships." in section
