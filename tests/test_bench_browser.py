import pytest
from PIL import Image, ImageDraw

from benchmarks import browser
from benchmarks.corpus import TextItem
from benchmarks.synthetic import assign_splits


def items(n=60):
    out = []
    for k in range(n):
        content = ("prose", "code", "ui")[k % 3]
        text = {"prose": f"Sentence {k} <b> & more", "code": f"def f{k}(x):\n    return x  # c",
                "ui": f"Open {k}\nSave {k}"}[content]  # fmt: skip
        out.append(TextItem(f"id{k:03d}", "en", content, text))
    return out


def fake_render(page, png, scale):
    image = Image.new("RGB", (int(808 * scale), int(300 * scale)), "#ffffff")
    box = [int(24 * scale), int(24 * scale), int(200 * scale), int(60 * scale)]
    ImageDraw.Draw(image).rectangle(box, fill="#000000")
    image.save(png)


def test_prose_is_escaped():
    page = browser.page_html(items()[0], "light", 13, "serif", "menu")
    assert "Sentence 0 &lt;b&gt; &amp; more" in page
    assert "font:13px/1.5 serif" in page


def test_code_keeps_its_lines_and_colours_tokens():
    page = browser.page_html(items()[1], "dark", 12, "monospace", "menu")
    assert '<span class="kw">def</span>' in page
    assert '<span class="com"># c</span>' in page
    assert "<pre>" in page and "\n" in page.split("<pre>")[1]


@pytest.mark.parametrize(
    "style, tag", [("buttons", 'class="button"'), ("menu", "<li>"), ("table", "<td>")]
)
def test_ui_puts_each_line_in_its_own_element(style, tag):
    page = browser.page_html(items()[2], "light", 14, "sans-serif", style)
    assert page.count(tag) == 2
    assert "Open 2" in page and "Save 2" in page


def test_trim_crops_to_the_content_plus_padding():
    image = Image.new("RGB", (300, 200), "#ffffff")
    ImageDraw.Draw(image).rectangle([100, 50, 149, 79], fill="#000000")
    assert browser.trim(image, 10).size == (70, 50)


def test_trim_clamps_the_padding_to_the_image():
    image = Image.new("RGB", (60, 40), "#ffffff")
    ImageDraw.Draw(image).rectangle([0, 0, 59, 39], fill="#000000")
    assert browser.trim(image, 10).size == (60, 40)


def test_trim_keeps_a_blank_image():
    assert browser.trim(Image.new("RGB", (30, 20), "#0d1117"), 5).size == (30, 20)


def test_generate_renders_test_texts_only(tmp_path):
    pool = items()
    samples = browser.generate(tmp_path, pool, seed=42, render=fake_render)
    splits = assign_splits(pool, 42)
    assert samples and len(samples) % browser.RENDERS_PER_TEXT == 0
    assert {splits[s.text_id] for s in samples} == {"test"}
    assert {(s.source, s.split, s.degradation) for s in samples} == {("browser", "browser", "none")}
    assert all(s.path.exists() and s.scale in browser.SCALES for s in samples)
    assert all(s.font == "monospace" for s in samples if s.content == "code")
    assert len({s.path.name for s in samples}) == len(samples)


def test_generate_is_deterministic(tmp_path):
    first = browser.generate(tmp_path / "a", items(), render=fake_render)
    second = browser.generate(tmp_path / "b", items(), render=fake_render)
    key = lambda s: (s.path.name, s.theme, s.font, s.font_size, s.scale)  # noqa: E731
    assert [key(s) for s in first] == [key(s) for s in second]


def test_missing_browser_is_reported(monkeypatch):
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Chrome or Chromium"):
        browser.find_chrome()
