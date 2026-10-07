import pytest
from PIL import Image

from sniptext.analyzer import ImageAnalyzer
from sniptext.pipelines import DEFAULT, MAX_ROUTED_PIXELS, STEPS, V04, Pipeline, recognize


def dark():
    return Image.new("RGB", (40, 20), (10, 10, 10))


class TestSteps:
    def test_light_inverts_dark_images_only(self):
        bright = Image.new("RGB", (40, 20), (240, 240, 240))
        assert STEPS["light"](dark()).getpixel((0, 0)) == (245, 245, 245)
        assert STEPS["light"](bright).getpixel((0, 0)) == (240, 240, 240)

    def test_light_returns_rgb_for_grayscale(self):
        out = STEPS["light"](Image.new("L", (40, 20), 10))
        assert out.mode == "RGB"
        assert out.getpixel((0, 0)) == (245, 245, 245)

    def test_up2_doubles_both_sides(self):
        assert STEPS["up2"](dark()).size == (80, 40)

    def test_median3_removes_a_single_pixel(self):
        image = Image.new("RGB", (9, 9), (255, 255, 255))
        image.putpixel((4, 4), (0, 0, 0))
        assert STEPS["median3"](image).getpixel((4, 4)) == (255, 255, 255)

    def test_gauss1_softens_an_edge_and_keeps_the_size(self):
        image = Image.new("L", (20, 10), 0)
        image.paste(255, (10, 0, 20, 10))
        out = STEPS["gauss1"](image)
        assert out.size == (20, 10)
        assert 0 < out.getpixel((10, 5)) < 255

    def test_enhance_is_the_v04_preprocessing(self):
        image = Image.new("RGB", (400, 120), (30, 30, 30))
        assert STEPS["enhance"](image).tobytes() == ImageAnalyzer().enhance_for_ocr(image).tobytes()


class TestPipeline:
    def test_prepare_applies_the_steps_in_order(self):
        out = Pipeline("x", ("light", "up2"), "6").prepare(dark())
        assert out.size == (80, 40)
        assert out.getpixel((0, 0)) == (245, 245, 245)

    def test_fixed_psm(self):
        assert Pipeline("x", ("light",), "6").psm_mode(dark()) == 6
        assert Pipeline("x", ("light",), "11").psm_mode(dark()) == 11

    def test_auto_psm_asks_the_analyzer(self):
        wide = Image.new("RGB", (900, 60), (255, 255, 255))
        assert Pipeline("x", ("light",), "auto").psm_mode(wide) == (
            ImageAnalyzer().suggest_psm_mode(wide)
        )

    @pytest.mark.parametrize("steps, psm", [(("sharpen",), "6"), (("light",), "3")])
    def test_unknown_step_or_psm_is_rejected(self, steps, psm):
        with pytest.raises(ValueError):
            Pipeline("x", steps, psm)

    def test_round_trip_through_a_dict(self):
        pipeline = Pipeline("light_up2_psm6", ("light", "up2"), "6")
        assert pipeline.to_dict() == {
            "name": "light_up2_psm6",
            "steps": ["light", "up2"],
            "psm": "6",
        }
        assert Pipeline.from_dict(pipeline.to_dict()) == pipeline

    def test_the_v04_pipeline(self):
        assert V04 == Pipeline("enhance_auto", ("enhance",), "auto")
        assert isinstance(DEFAULT, Pipeline)
        assert MAX_ROUTED_PIXELS == 2_000_000


def tesseract_data(words):
    """image_to_data output for (block, line, word_num, text, conf) tuples."""
    keys = ("block_num", "par_num", "line_num", "word_num", "text", "conf")
    columns = {key: [] for key in keys}
    for block, line, number, text, conf in words:
        for key, value in zip(keys, (block, 1, line, number, text, conf)):
            columns[key].append(value)
    return columns


class TestRecognize:
    def test_groups_words_into_lines_with_confidences(self, monkeypatch):
        calls = []

        def fake(image, lang, config, output_type):
            calls.append((image.size, lang, config))
            return tesseract_data([(1, 1, 2, "cd", 80), (1, 1, 1, "ab", 90), (1, 2, 1, "ef", 50)])

        monkeypatch.setattr("pytesseract.image_to_data", fake)
        text, confs = recognize(Pipeline("x", ("light", "up2"), "6"), dark(), "eng+rus")
        assert text == "ab cd\nef"
        assert confs == [[0.9, 0.8], [0.5]]
        assert calls == [((80, 40), "eng+rus", "--oem 1 --psm 6")]

    def test_blank_words_and_negative_confidences_are_skipped(self, monkeypatch):
        data = tesseract_data([(1, 1, 1, "  ", 95), (1, 1, 2, "ok", -1), (1, 1, 3, "yes", 70)])
        monkeypatch.setattr("pytesseract.image_to_data", lambda *a, **k: data)
        assert recognize(V04, Image.new("RGB", (400, 120), "white"), "eng") == ("yes", [[0.7]])

    def test_no_words_gives_empty_text(self, monkeypatch):
        monkeypatch.setattr("pytesseract.image_to_data", lambda *a, **k: tesseract_data([]))
        assert recognize(V04, Image.new("RGB", (400, 120), "white"), "eng") == ("", [])
