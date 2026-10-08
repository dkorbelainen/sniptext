import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from sniptext.config import Config
from sniptext.ocr import OCREngine, OCRError
from sniptext.pipelines import LARGE_IMAGE, MAX_ROUTED_PIXELS, Pipeline

FIRST = Pipeline("first", ("light",), "6")
OTHER = Pipeline("other", ("light", "up2"), "6")


class FakeRouter:
    def __init__(self, policy="pre_ocr", pick=OTHER, available=True):
        self.policy, self.pick, self.available = policy, pick, available
        self.actions, self.default = (FIRST, OTHER), FIRST
        self.calls = []

    def choose(self, features, first_conf=None):
        self.calls.append(first_conf)
        return self.pick


@pytest.fixture(autouse=True)
def installed_languages(monkeypatch):
    monkeypatch.setattr("pytesseract.get_languages", lambda config="": ["eng", "osd", "rus"])


@pytest.fixture
def make(monkeypatch):
    """Build an engine whose Tesseract passes return the given text per pipeline name."""
    monkeypatch.setattr("pytesseract.get_tesseract_version", lambda: "5.0")

    def build(texts, router=None, **config):
        engine = OCREngine(Config(**config))
        engine.router = router or FakeRouter()
        passes = []

        def fake(pipeline, image, lang):
            passes.append(pipeline.name)
            result = texts[pipeline.name]
            if isinstance(result, Exception):
                raise result
            return result, [[0.8]]

        monkeypatch.setattr("sniptext.ocr.recognize", fake)
        return engine, passes

    return build


def white(size=(200, 60)):
    return Image.new("RGB", size, "white")


class TestPrepareImage:
    def engine(self, monkeypatch, **config):
        monkeypatch.setattr("pytesseract.get_tesseract_version", lambda: "5.0")
        return OCREngine(Config(**config))

    def test_numpy_arrays(self, monkeypatch):
        engine = self.engine(monkeypatch)
        assert engine._prepare_image(np.zeros((10, 20), dtype=np.uint8)).mode == "L"
        assert engine._prepare_image(np.zeros((10, 20, 3), dtype=np.uint8)).mode == "RGB"
        assert engine._prepare_image(np.zeros((10, 20, 4), dtype=np.uint8)).mode == "RGB"

    def test_transparent_pixels_become_white(self, monkeypatch):
        image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        image.putpixel((5, 5), (0, 0, 0, 255))
        out = self.engine(monkeypatch)._prepare_image(image)
        assert out.mode == "RGB"
        assert out.getpixel((0, 0)) == (255, 255, 255) and out.getpixel((5, 5)) == (0, 0, 0)

    def test_light_text_on_transparency_gets_a_dark_backdrop(self, monkeypatch):
        image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        image.putpixel((5, 5), (255, 255, 255, 255))
        out = self.engine(monkeypatch)._prepare_image(image)
        assert out.getpixel((0, 0)) == (0, 0, 0) and out.getpixel((5, 5)) == (255, 255, 255)

    def test_a_fully_transparent_image_becomes_white(self, monkeypatch):
        out = self.engine(monkeypatch)._prepare_image(Image.new("RGBA", (20, 10), (9, 9, 9, 0)))
        assert out.getpixel((0, 0)) == (255, 255, 255)

    def test_palette_images_keep_their_colours(self, monkeypatch):
        image = Image.new("RGB", (20, 10), (200, 30, 30)).convert("P")
        out = self.engine(monkeypatch)._prepare_image(image)
        assert out.mode == "RGB" and out.getpixel((0, 0))[0] > 150

    def test_palette_transparency_becomes_white(self, monkeypatch):
        image = Image.new("P", (20, 10), 0)
        image.putpalette([0, 0, 0] * 256)
        image.info["transparency"] = 0
        out = self.engine(monkeypatch)._prepare_image(image)
        assert out.getpixel((0, 0)) == (255, 255, 255)

    def test_large_images_are_reduced_to_the_configured_side(self, monkeypatch):
        out = self.engine(monkeypatch, max_image_size=100)._prepare_image(white((400, 200)))
        assert out.size == (100, 50)

    def test_the_caller_image_is_not_modified(self, monkeypatch):
        image = white((400, 200))
        self.engine(monkeypatch, max_image_size=100)._prepare_image(image)
        assert image.size == (400, 200)


class TestBeforeOcr:
    def test_runs_the_chosen_pipeline_once(self, make):
        engine, passes = make({"first": "a", "other": "b"})
        assert engine.recognize(white()) == "b"
        assert passes == ["other"]

    def test_default_choice_runs_the_default(self, make):
        engine, passes = make({"first": "a", "other": "b"}, FakeRouter(pick=FIRST))
        assert engine.recognize(white()) == "a"
        assert passes == ["first"]

    def test_empty_result_falls_back_to_the_default(self, make):
        engine, passes = make({"first": "a", "other": ""})
        assert engine.recognize(white()) == "a"
        assert passes == ["other", "first"]

    def test_failing_pipeline_falls_back_to_the_default(self, make):
        engine, passes = make({"first": "a", "other": RuntimeError("boom")})
        assert engine.recognize(white()) == "a"
        assert passes == ["other", "first"]

    def test_accepts_numpy_input(self, make):
        engine, _ = make({"first": "a", "other": "b"})
        assert engine.recognize(np.full((60, 200, 3), 255, dtype=np.uint8)) == "b"


class TestCascade:
    def test_first_pass_confidences_reach_the_router(self, make):
        router = FakeRouter(policy="cascade")
        engine, passes = make({"first": "a", "other": "b"}, router)
        assert engine.recognize(white()) == "b"
        assert passes == ["first", "other"] and router.calls == [[[0.8]]]

    def test_keeping_the_first_pass_costs_one_pass(self, make):
        engine, passes = make({"first": "a", "other": "b"}, FakeRouter("cascade", pick=FIRST))
        assert engine.recognize(white()) == "a"
        assert passes == ["first"]

    def test_empty_second_pass_keeps_the_first_text(self, make):
        engine, passes = make({"first": "a", "other": ""}, FakeRouter("cascade"))
        assert engine.recognize(white()) == "a"
        assert passes == ["first", "other"]


class TestNotRouted:
    def test_routing_switched_off_in_the_config(self, make):
        router = FakeRouter()
        engine, passes = make({"first": "a", "other": "b"}, router, routing=False)
        assert engine.recognize(white()) == "a"
        assert passes == ["first"] and router.calls == []

    def test_router_without_a_model(self, make):
        router = FakeRouter(available=False)
        engine, passes = make({"first": "a", "other": "b"}, router)
        assert engine.recognize(white()) == "a"
        assert router.calls == []

    def test_an_image_above_the_size_limit_runs_the_large_image_pipeline(self, make):
        router = FakeRouter()
        engine, passes = make({"first": "a", "other": "b", LARGE_IMAGE.name: "c"}, router)
        side = int(MAX_ROUTED_PIXELS**0.5) + 10
        assert engine.recognize(white((side, side))) == "c"
        assert passes == [LARGE_IMAGE.name] and router.calls == []

    def test_an_image_above_the_size_limit_with_routing_off(self, make):
        engine, passes = make({"first": "a", LARGE_IMAGE.name: "c"}, routing=False)
        side = int(MAX_ROUTED_PIXELS**0.5) + 10
        assert engine.recognize(white((side, side))) == "c"
        assert passes == [LARGE_IMAGE.name]

    def test_large_image_pipeline_failure_is_reported(self, make):
        engine, _ = make({"first": "a", LARGE_IMAGE.name: RuntimeError("boom")})
        side = int(MAX_ROUTED_PIXELS**0.5) + 10
        with pytest.raises(OCRError, match="boom"):
            engine.recognize(white((side, side)))

    def test_feature_extraction_failure(self, make, monkeypatch):
        engine, passes = make({"first": "a", "other": "b"})
        monkeypatch.setattr(engine._analyzer, "extract_features", lambda image: 1 / 0)
        assert engine.recognize(white()) == "a"
        assert passes == ["first"]


class TestLanguages:
    def engine(self, monkeypatch, language, installed):
        monkeypatch.setattr("pytesseract.get_tesseract_version", lambda: "5.0")
        monkeypatch.setattr("pytesseract.get_languages", installed)
        return OCREngine(Config(ocr_language=language))

    def test_a_missing_language_pack_is_reported(self, monkeypatch):
        # Tesseract itself drops the missing language and reads garbage with the rest.
        with pytest.raises(OCRError, match="rus, ell") as error:
            self.engine(monkeypatch, "eng+rus+ell", lambda config="": ["eng", "osd"])
        assert "tesseract-data-rus" in str(error.value)

    def test_installed_languages_pass(self, monkeypatch):
        self.engine(monkeypatch, "eng+rus", lambda config="": ["eng", "osd", "rus"])

    def test_an_unreadable_language_list_does_not_block(self, monkeypatch):
        def broken(config=""):
            raise RuntimeError("no list")

        self.engine(monkeypatch, "eng+rus", broken)


class TestFailures:
    def test_tesseract_missing(self, monkeypatch):
        def missing():
            raise FileNotFoundError("tesseract is not installed")

        monkeypatch.setattr("pytesseract.get_tesseract_version", missing)
        with pytest.raises(OCRError, match="Tesseract is not available"):
            OCREngine(Config())

    def test_default_pipeline_failure_is_reported_not_swallowed(self, make):
        error = RuntimeError("Failed loading language 'ell'")
        engine, _ = make({"first": error, "other": "b"}, FakeRouter(pick=FIRST))
        with pytest.raises(OCRError, match="Failed loading language 'ell'"):
            engine.recognize(white())

    def test_zero_size_image_is_empty_without_running_tesseract(self, make):
        engine, passes = make({"first": "a", "other": "b"})
        assert engine.recognize(np.zeros((0, 0, 3), dtype=np.uint8)) == ""
        assert passes == []

    @pytest.mark.parametrize("size", [(1, 1), (200, 60)])
    def test_blank_image_is_empty_after_at_most_two_passes(self, make, size):
        engine, passes = make({"first": "", "other": ""})
        assert engine.recognize(white(size)) == ""
        assert len(passes) <= 2


def test_shipped_router_with_a_real_config(monkeypatch):
    monkeypatch.setattr("pytesseract.get_tesseract_version", lambda: "5.0")
    passes = []
    monkeypatch.setattr(
        "sniptext.ocr.recognize", lambda p, image, lang: (passes.append(p.name), ("x", [[0.9]]))[1]
    )
    engine = OCREngine(Config(ocr_language="eng+rus"))
    assert engine.recognize(white((400, 120))) == "x"
    assert 1 <= len(passes) <= 2
    assert passes[0] in {action.name for action in engine.router.actions}


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs the tesseract binary")
def test_real_tesseract_reads_rendered_text():
    image = Image.new("RGB", (520, 90), "white")
    font = ImageFont.load_default(size=40)
    ImageDraw.Draw(image).text((20, 20), "Hello world 2026", fill="black", font=font)
    assert OCREngine(Config()).recognize(image) == "Hello world 2026"
