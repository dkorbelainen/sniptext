"""Benchmark plumbing that needs no OCR: pool runner, legacy baseline, row building."""

import json

import numpy as np
import pytest
from benchmarks.pool import POOL
from PIL import Image

from benchmarks import run_eval
from benchmarks.evaluate import (
    cluster_bootstrap,
    realized,
    summarize,
    time_matrix,
    weak_label_agreement,
)
from benchmarks.legacy_policy import legacy_actions, legacy_features
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


def test_legacy_features_keep_the_old_seven():
    assert legacy_features(Image.new("RGB", (300, 100), (240, 240, 240))).shape == (7,)


def test_legacy_actions_follow_the_old_rules():
    # brightness, contrast, sharpness, has_color, size_ratio, text_density, noise_level
    clear = [0.8, 0.9, 0.9, 0.0, 0.5, 0.2, 0.05]
    noisy = [0.5, 0.4, 0.5, 0.0, 0.5, 0.2, 0.9]
    empty = [0.5, 0.6, 0.6, 0.0, 0.5, 0.0, 0.1]
    actions = legacy_actions(np.array([clear, noisy, empty]))
    assert actions.tolist() == [0, 2, 2]


def test_legacy_actions_are_deterministic_for_a_seed():
    borderline = np.random.default_rng(0).uniform(0.3, 0.5, size=(40, 7))
    assert (
        legacy_actions(borderline, seed=3).tolist() == legacy_actions(borderline, seed=3).tolist()
    )
    assert set(legacy_actions(borderline, seed=3).tolist()) <= {0, 2}


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


def test_realized_and_time_matrix():
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert realized(matrix, np.array([2, 0])).tolist() == [3.0, 4.0]
    t_tess, t_easy = np.array([0.1, 0.2]), np.array([0.4, 0.5])
    assert time_matrix("pre_ocr", t_tess, t_easy) == pytest.approx(
        np.array([[0.1, 0.4, 0.5], [0.2, 0.5, 0.7]])
    )
    assert time_matrix("cascade", t_tess, t_easy) == pytest.approx(
        np.array([[0.1, 0.5, 0.5], [0.2, 0.7, 0.7]])
    )


def test_summarize():
    cer = np.array([[0.0, 0.5, 0.2], [2.0, 0.1, 0.3], [0.4, 0.4, 0.0], [0.3, 0.9, 0.6]])
    times = time_matrix("pre_ocr", np.full(4, 0.1), np.full(4, 0.4))
    clusters = np.array(["a", "a", "b", "c"])
    oracle = summarize("oracle", cer, np.minimum(cer, 1.0).argmin(axis=1), times, clusters)
    assert oracle["regret"][0] == 0.0
    assert oracle["cer"][0] == pytest.approx(0.1)
    always = summarize("always_tesseract", cer, np.zeros(4, dtype=int), times, clusters,
                       reference=np.minimum(cer[:, 1], 1.0))  # fmt: skip
    assert always["n"] == 4
    assert always["cer"][0] == pytest.approx((0.0 + 1.0 + 0.4 + 0.3) / 4)
    assert always["cer_unclipped"] == pytest.approx((0.0 + 2.0 + 0.4 + 0.3) / 4)
    assert always["cer_median"] == pytest.approx(0.35)
    assert always["share"] == {"tesseract": 1.0, "easyocr": 0.0, "merge": 0.0}
    assert always["time"] == pytest.approx(0.1)
    assert always["delta"][0] == pytest.approx((1.7 - 1.9) / 4)
    assert "delta" not in oracle


def weak_row(fast, ens, cer_tess, cer_merge):
    return {
        "weak": {"fast_quality": fast, "ens_quality": ens},
        "cer": {"tesseract_plain": cer_tess, "merge": cer_merge},
    }


def test_weak_label_agreement_perfect_and_useless():
    perfect = [weak_row(0.9, 0.5, 0.0, 0.3), weak_row(0.4, 0.8, 0.5, 0.1)] * 10
    result = weak_label_agreement(perfect)
    assert result["accuracy"] == 1.0 and result["kappa"] == pytest.approx(1.0)
    assert result["recorded_share"] == 1.0
    always_merge = [weak_row(0.4, 0.8, 0.0, 0.3), weak_row(0.4, 0.8, 0.5, 0.1)] * 10
    result = weak_label_agreement(always_merge)
    assert result["weak_merge_share"] == 1.0
    assert result["true_merge_share"] == 0.5
    assert result["kappa"] == 0.0
