"""Report rendering from a fabricated evaluation (no OCR, no plotting)."""

import json
import re

from test_bench_train import RIDGE_ONLY, fake_results

from benchmarks import report, train_router

CALIB = {
    "synthetic": {"n": 100, "accuracy": 0.6, "mean_conf": 0.7, "ece_raw": 0.1, "ece_cal": 0.02}
}
MERGE = {
    "synthetic": {"n_test": 30, "cer_heuristic": 0.3, "cer_rawconf": 0.2, "cer_calibrated": 0.19}
}


def evaluation(tmp_path):
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"n": 5, "easyocr_cpu_mean": 3.0, "easyocr_gpu_mean": 0.3}))
    return train_router.run(
        fake_results(tmp_path / "r.json"),
        tmp_path / "t.csv.gz",
        tmp_path / "eval.json",
        timing,
        candidates=RIDGE_ONLY,
    )


def test_render_has_every_section_and_no_missing_numbers(tmp_path):
    text = report.render(evaluation(tmp_path), CALIB, MERGE, "abc1234", "2026-10-07")
    for heading in (
        "# OCR engine router benchmark",
        "## Task",
        "## Data",
        "## Result on held-out texts",
        "## Unseen fonts",
        "## Out-of-domain receipts",
        "## Accuracy against time",
        "## Model selection",
        "## Features",
        "## Breakdown",
        "## The previous selector",
        "## Confidence calibration",
        "## Limitations",
        "## Reproduce",
    ):
        assert heading in text, heading
    assert re.search(r"\bnan\b", text.lower()) is None
    assert "abc1234" in text
    assert "Oracle (lower bound)" in text
    assert "![" in text


def test_render_gives_a_verdict_per_slice_and_the_shipping_rule(tmp_path):
    ev = evaluation(tmp_path)
    text = report.render(ev, CALIB, MERGE, "abc1234", "2026-10-07")
    assert text.count("95% interval over texts") == len(ev["slices"])
    assert "out-of-fold CER" in text and "ships" in text
    assert "spans" in text


def test_render_without_cpu_timing(tmp_path):
    ev = evaluation(tmp_path)
    ev["cpu"] = None
    text = report.render(ev, CALIB, MERGE, "abc1234", "2026-10-07")
    assert "CPU timing was not measured" in text


def summary(delta):
    return {"policy": "router_cascade", "cer": (0.2, 0.18, 0.22), "delta": delta}


def test_verdict_states_what_the_interval_supports():
    assert "lower than" in report.verdict(summary((-0.05, -0.08, -0.02)), "easyocr")
    assert "higher than" in report.verdict(summary((0.05, 0.02, 0.08)), "easyocr")
    assert "not distinguishable" in report.verdict(summary((-0.01, -0.03, 0.01)), "easyocr")
    assert "EasyOCR" in report.verdict(summary((-0.05, -0.08, -0.02)), "easyocr")
