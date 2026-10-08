"""Synthetic corpus generator (Pillow's built-in font, no system fonts)."""

import random

import pytest
from PIL import Image, ImageFont

from benchmarks import synthetic
from benchmarks.corpus import TextItem


@pytest.fixture(autouse=True)
def builtin_font(monkeypatch):
    monkeypatch.setattr(synthetic, "_load_font", lambda family, size: ImageFont.load_default(size))


def items(n=40):
    out = []
    for i in range(n):
        lang = "ru" if i % 2 else "en"
        content = ("prose", "code", "ui")[i % 3]
        out.append(TextItem(f"id{i:03d}", lang, content, f"sample text {i}\nsecond line {i}"))
    return out


def test_splits_are_disjoint_and_proportional():
    pool = items(200)
    splits = synthetic.assign_splits(pool, seed=42)
    assert set(splits) == {item.text_id for item in pool}
    counts = {name: list(splits.values()).count(name) for name in ("train", "val", "test")}
    assert sum(counts.values()) == 200
    assert 110 <= counts["train"] <= 130
    assert 30 <= counts["val"] <= 50
    assert 30 <= counts["test"] <= 50


def test_splits_are_stratified():
    pool = items(120)
    splits = synthetic.assign_splits(pool, seed=42)
    for lang in ("en", "ru"):
        group = [splits[i.text_id] for i in pool if i.lang == lang]
        assert {"train", "val", "test"} <= set(group)


def test_generate_counts_and_metadata(tmp_path):
    pool = items(20)
    samples = synthetic.generate(tmp_path, pool, seed=42)
    splits = synthetic.assign_splits(pool, seed=42)
    n_test = list(splits.values()).count("test")
    assert len(samples) == 20 * 4 + n_test * len(synthetic.UNSEEN_FONTS)
    for s in samples:
        assert s.path.exists()
        assert s.source == "synthetic"
        assert 11 <= s.font_size <= 30
        assert s.theme in synthetic.THEMES
        if s.split == "unseen_font":
            assert s.font in synthetic.UNSEEN_FONTS
            assert splits[s.text_id] == "test"
        else:
            assert s.font in synthetic.SEEN_FONTS
            assert s.split == splits[s.text_id]


def test_generate_is_deterministic(tmp_path):
    a = synthetic.generate(tmp_path / "a", items(6), seed=7)
    b = synthetic.generate(tmp_path / "b", items(6), seed=7)
    assert [(s.font, s.font_size, s.theme, s.degradation) for s in a] == [
        (s.font, s.font_size, s.theme, s.degradation) for s in b
    ]
    for sa, sb in zip(a, b):
        assert sa.path.read_bytes() == sb.path.read_bytes()


def test_every_degradation_keeps_size_and_mode():
    img = synthetic.render("hello world\nsecond line", "any", 18, "dark")
    for name, degrade in synthetic.DEGRADATIONS.items():
        out = degrade(img, random.Random(0))
        assert isinstance(out, Image.Image), name
        assert out.size == img.size, name
        assert out.mode == "RGB", name


def test_degradation_names_cover_the_spec():
    assert set(synthetic.DEGRADATIONS) == {"blur", "noise", "jpeg", "rescale", "lowcontrast"}


def test_missing_font_family_is_an_error(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(
        synthetic.subprocess, "check_output", lambda *a, **k: b"DejaVu Sans|/usr/share/x.ttf"
    )
    synthetic._font_file.cache_clear()
    with pytest.raises(RuntimeError, match="No Such Family"):
        synthetic._font_file("No Such Family")
    synthetic._font_file.cache_clear()


def pool(prefix, n=40):
    out = []
    for k in range(n):
        lang, content = (("en", "prose"), ("ru", "prose"), ("en", "code"), ("en", "ui"))[k % 4]
        words = " ".join(f"{prefix}{k}w{j}" for j in range(8))
        out.append(TextItem(f"{prefix}{k:03d}", lang, content, words))
    return out


def test_fresh_items_share_no_word_run_with_used_texts():
    used = pool("u")
    overlapping = TextItem("x001", "en", "prose", "intro " + used[0].text + " outro")
    same_id = TextItem(used[1].text_id, "en", "prose", "entirely different words here ok fine yes")
    candidates = pool("c") + [overlapping, same_id]
    fresh = synthetic.fresh_items(candidates, used, 12, seed=1)
    ids = {item.text_id for item in fresh}
    assert len(fresh) == 12 and "x001" not in ids and used[1].text_id not in ids
    assert ids <= {item.text_id for item in candidates}


def test_fresh_items_follow_the_used_pool_proportions_and_are_deterministic():
    used, candidates = pool("u"), pool("c", 80)
    fresh = synthetic.fresh_items(candidates, used, 20, seed=1)
    counts = {}
    for item in fresh:
        counts[(item.lang, item.content)] = counts.get((item.lang, item.content), 0) + 1
    assert counts == {("en", "prose"): 5, ("ru", "prose"): 5, ("en", "code"): 5, ("en", "ui"): 5}
    assert fresh == synthetic.fresh_items(candidates, used, 20, seed=1)
    assert fresh != synthetic.fresh_items(candidates, used, 20, seed=2)


def test_fresh_items_fail_when_a_group_runs_short():
    with pytest.raises(ValueError, match="fresh"):
        synthetic.fresh_items(pool("c", 8), pool("u"), 20, seed=1)


def test_generate_slice_renders_every_text_under_one_split(tmp_path):
    texts = pool("c", 6)
    samples = synthetic.generate_slice(tmp_path, texts, "confirm", seed=7)
    assert len(samples) == 6 * synthetic.RENDERS_PER_TEXT
    assert {s.split for s in samples} == {"confirm"} and {s.source for s in samples} == {
        "synthetic"
    }
    assert {s.font for s in samples} <= set(synthetic.SEEN_FONTS)
    assert all(s.path.exists() for s in samples)
    again = synthetic.generate_slice(tmp_path / "b", texts, "confirm", seed=7)
    assert [(s.font, s.font_size, s.theme, s.degradation) for s in samples] == [
        (s.font, s.font_size, s.theme, s.degradation) for s in again
    ]
