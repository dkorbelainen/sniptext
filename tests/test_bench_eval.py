"""Benchmark plumbing that needs no OCR: pool runner, row building, evaluation helpers."""

import json

import numpy as np
import pytest
from PIL import Image

from benchmarks import run_eval
from benchmarks.evaluate import cluster_bootstrap, effective, paired, realized, summarize
from benchmarks.pool import POOL
from sniptext.analyzer import FEATURE_NAMES, ImageAnalyzer
from sniptext.pipelines import V04, Pipeline


def test_pool_has_thirteen_distinct_pipelines_including_the_v04_one():
    assert len(POOL) == 13
    assert len({p.name for p in POOL}) == 13
    assert len({(p.steps, p.psm) for p in POOL}) == 13
    assert V04 in POOL
    assert Pipeline("light_psm11", ("light",), "11") in POOL
    assert Pipeline("light_up2_median3_psm6", ("light", "up2", "median3"), "6") in POOL


def test_run_pool_runs_each_pipeline_once(monkeypatch):
    seen = []

    def fake(pipeline, image, lang):
        seen.append((pipeline.name, lang))
        return pipeline.name, [[0.5]]

    monkeypatch.setattr(run_eval, "recognize", fake)
    outputs = run_eval.run_pool(Image.new("RGB", (40, 20), "white"), POOL[:3], "eng")
    assert seen == [(p.name, "eng") for p in POOL[:3]]
    assert outputs[POOL[0].name] == (POOL[0].name, [[0.5]])


def test_run_pool_records_a_failing_pipeline_as_empty(monkeypatch):
    def fake(pipeline, image, lang):
        if pipeline is POOL[1]:
            raise RuntimeError("tesseract crashed")
        return "ok", [[0.9]]

    monkeypatch.setattr(run_eval, "recognize", fake)
    outputs = run_eval.run_pool(Image.new("RGB", (40, 20), "white"), POOL[:2], "eng")
    assert outputs[POOL[1].name] == ("", [])
    assert outputs[POOL[0].name] == ("ok", [[0.9]])


def sample_dict(path):
    return {"path": path, "gt": "hello  world", "source": "synthetic", "split": "val",
            "text_id": "t1", "lang": "en", "content": "prose", "font": "F", "font_size": 14,
            "theme": "light", "degradation": "noise", "scale": 1.0}  # fmt: skip


def test_build_row(tmp_path):
    image = Image.new("RGB", (120, 40), "white")
    outputs = {
        "a": ("hello world", [[0.9, 0.5]]),
        "b": ("hellp world", [[0.4, 0.4]]),
        "c": ("", []),
    }
    row = run_eval.build_row(sample_dict(tmp_path / "x.png"), image, outputs, ImageAnalyzer())
    assert row["image"] == "x.png" and row["split"] == "val" and row["pixels"] == 4800
    assert row["gt"] == "hello  world" and row["scale"] == 1.0
    assert len(row["features"]) == len(FEATURE_NAMES)
    assert row["cer"]["a"] == 0.0 and 0 < row["cer"]["b"] < 0.2 and row["cer"]["c"] == 1.0
    assert row["text"] == {"a": "hello world", "b": "hellp world", "c": ""}
    assert row["conf_stats"]["a"][0] == 0.7 and row["conf_stats"]["c"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    json.dumps(row)


def test_timing_images_takes_a_dev_sample_and_every_eval_image():
    samples = [{"path": f"d{k}.png", "split": "train" if k % 2 else "val"} for k in range(50)]
    samples += [{"path": f"e{k}.png", "split": s} for k, s in enumerate(run_eval.EVAL_SLICES * 3)]
    chosen = run_eval.timing_images(samples, seed=1, n_dev=10)
    assert sum(s["split"] in run_eval.DEV_SPLITS for s in chosen) == 10
    assert sum(s["split"] in run_eval.EVAL_SLICES for s in chosen) == 12
    assert chosen == run_eval.timing_images(samples, seed=1, n_dev=10)


def test_collect_samples_adds_the_browser_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{**sample_dict("b.png"), "source": "browser", "split": "browser"}])
    )
    monkeypatch.setattr(run_eval, "generate", lambda out, items, seed: [])
    monkeypatch.setattr(run_eval, "load_items", lambda seed: [])
    monkeypatch.setattr(
        run_eval, "load_sroie", lambda limit: iter([(tmp_path / "r.jpg", "TOTAL 5")])
    )
    samples = run_eval.collect_samples(42, 1, manifest)
    assert [s["split"] for s in samples] == ["ood", "browser"]
    assert samples[0]["scale"] == 1.0 and samples[0]["gt"] == "TOTAL 5"
    assert run_eval.collect_samples(42, 1, tmp_path / "missing.json")[-1]["split"] == "ood"


def test_bootstrap_of_a_constant_is_that_constant():
    assert cluster_bootstrap([0.3] * 12, list(range(12))) == pytest.approx((0.3, 0.3, 0.3))


def test_bootstrap_interval_brackets_the_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    values = rng.random(300)
    clusters = np.arange(300)
    mean, low, high = cluster_bootstrap(values, clusters)
    assert low < mean < high
    assert cluster_bootstrap(values, clusters) == (mean, low, high)
    assert cluster_bootstrap(values, clusters, seed=5) != (mean, low, high)


def test_clustered_values_give_a_wider_interval():
    rng = np.random.default_rng(2)
    per_cluster = rng.random(50)
    values = np.repeat(per_cluster, 6)
    _, low_c, high_c = cluster_bootstrap(values, np.repeat(np.arange(50), 6))
    _, low_i, high_i = cluster_bootstrap(values, np.arange(300))
    assert (high_c - low_c) > 1.5 * (high_i - low_i)


def test_realized_picks_the_chosen_column():
    assert realized([[1, 2], [3, 4]], [1, 0]).tolist() == [2, 3]


def test_effective_pre_ocr_falls_back_to_the_default_on_empty_text():
    cer = np.array([[0.2, 1.0], [0.3, 0.1]])
    seconds = np.array([[0.1, 0.4], [0.1, 0.4]])
    empty = np.array([[False, True], [True, False]])
    out_cer, out_seconds = effective("pre_ocr", cer, seconds, empty)
    # row 0: the second action came back empty, so the default ran as well
    assert out_cer.tolist() == [[0.2, 0.2], [0.3, 0.1]]
    assert out_seconds.tolist() == [[0.1, 0.5], [0.1, 0.4]]


def test_effective_cascade_always_pays_for_the_first_pass():
    cer = np.array([[0.2, 1.0], [0.3, 0.1]])
    seconds = np.array([[0.1, 0.4], [0.1, 0.4]])
    empty = np.array([[False, True], [False, False]])
    out_cer, out_seconds = effective("cascade", cer, seconds, empty)
    assert out_cer.tolist() == [[0.2, 0.2], [0.3, 0.1]]
    assert out_seconds.tolist() == [[0.1, 0.5], [0.1, 0.5]]


def test_effective_leaves_its_inputs_alone_and_rejects_other_policies():
    cer, seconds = np.array([[0.2, 1.0]]), np.array([[0.1, 0.4]])
    effective("cascade", cer, seconds, np.array([[False, True]]))
    assert cer.tolist() == [[0.2, 1.0]] and seconds.tolist() == [[0.1, 0.4]]
    with pytest.raises(ValueError):
        effective("static", cer, seconds, np.array([[False, False]]))


def test_summarize():
    cer = np.array([0.0, 2.0, 0.5, 0.5])
    out = summarize("p", cer, [0.1, 0.3, 0.2, 0.2], ["a", "a", "b", "b"],
                    oracle=np.zeros(4), reference=np.full(4, 0.25), share={"x": 1.0})  # fmt: skip
    assert out["policy"] == "p" and out["n"] == 4
    assert out["cer"][0] == pytest.approx(0.5)  # 2.0 is clipped to 1
    assert out["cer_unclipped"] == pytest.approx(0.75) and out["cer_median"] == 0.5
    assert out["time"] == pytest.approx(0.2)
    assert out["regret"][0] == pytest.approx(0.5) and out["delta"][0] == pytest.approx(0.25)
    assert out["share"] == {"x": 1.0}
    assert "delta" not in summarize("q", cer, [0.1] * 4, ["a", "a", "b", "b"])


def test_paired_difference_clips_both_sides():
    mean, low, high = paired([3.0, 0.0], [0.5, 0.0], ["a", "b"])
    assert mean == pytest.approx(0.25) and low <= mean <= high
