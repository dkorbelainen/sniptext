"""Tests for OCREngine internals (no real OCR calls)."""

import numpy as np
from PIL import Image

from sniptext.config import Config
from sniptext.ocr import EasyOCRBackend, OCREngine, TesseractBackend


class TestPrepareImage:
    """Test OCREngine._prepare_image."""

    def setup_method(self):
        self.config = Config()
        self.engine = OCREngine(self.config)

    def test_pil_image_passthrough(self):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        result = self.engine._prepare_image(img)
        assert isinstance(result, Image.Image)

    def test_rgb_numpy_array(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.engine._prepare_image(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_grayscale_numpy_array(self):
        arr = np.zeros((100, 100), dtype=np.uint8)
        result = self.engine._prepare_image(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "L"

    def test_rgba_numpy_array(self):
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        result = self.engine._prepare_image(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    def test_max_image_size_respected(self):
        config = Config(max_image_size=64)
        engine = OCREngine(config)
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        result = engine._prepare_image(arr)
        assert max(result.width, result.height) <= 64

    def test_small_image_not_resized(self):
        config = Config(max_image_size=4096)
        engine = OCREngine(config)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = engine._prepare_image(arr)
        assert result.width == 100
        assert result.height == 100

    def test_pil_image_not_mutated_on_resize(self):
        config = Config(max_image_size=64)
        engine = OCREngine(config)
        original = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
        engine._prepare_image(original)
        assert original.width == 200
        assert original.height == 200


    def test_returns_list(self):
        engine = OCREngine(Config())
        backends = engine.get_available_backends()
        assert isinstance(backends, list)

    def test_tesseract_in_backends(self):
        engine = OCREngine(Config())
        backends = engine.get_available_backends()
        assert "tesseract" in backends


class TestEasyOCRBackendLangCodes:
    def test_eng_maps_to_en(self):
        config = Config(ocr_language="eng")
        backend = EasyOCRBackend(config)
        assert backend._get_lang_codes() == ["en"]

    def test_rus_maps_to_ru(self):
        config = Config(ocr_language="rus")
        backend = EasyOCRBackend(config)
        assert backend._get_lang_codes() == ["ru"]

    def test_multi_language(self):
        config = Config(ocr_language="eng+rus")
        backend = EasyOCRBackend(config)
        codes = backend._get_lang_codes()
        assert "en" in codes
        assert "ru" in codes

    def test_unknown_lang_passed_as_is(self):
        config = Config(ocr_language="xyz")
        backend = EasyOCRBackend(config)
        assert backend._get_lang_codes() == ["xyz"]


class TestTesseractBackendLangCode:
    def test_returns_config_language(self):
        config = Config(ocr_language="eng")
        backend = TesseractBackend(config)
        assert backend._get_lang_code() == "eng"

    def test_returns_custom_language(self):
        config = Config(ocr_language="rus")
        backend = TesseractBackend(config)
        assert backend._get_lang_code() == "rus"
