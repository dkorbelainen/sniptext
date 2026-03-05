"""Tests for OCREngine internals (no real OCR calls)."""

from unittest.mock import MagicMock, patch

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

    def test_rgba_numpy_array_converted_to_rgb(self):
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        result = self.engine._prepare_image(arr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_rgba_pil_image_converted_to_rgb(self):
        img = Image.fromarray(np.zeros((100, 100, 4), dtype=np.uint8), mode="RGBA")
        result = self.engine._prepare_image(img)
        assert result.mode == "RGB"

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


class TestGetAvailableBackends:
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

    def test_confidence_threshold_respected(self):
        """Detections below ocr_confidence_threshold must be dropped."""
        config = Config(ocr_language="eng", ocr_confidence_threshold=0.7)
        backend = EasyOCRBackend(config)
        backend._available = True
        backend._initialized = True

        mock_reader = MagicMock()
        # Two detections: one above threshold, one below
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "hello", 0.9),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "noise", 0.4),
        ]
        backend._reader = mock_reader

        img = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
        result = backend.recognize(img)
        assert "hello" in result
        assert "noise" not in result


class TestTesseractBackendLangCode:
    def test_returns_config_language(self):
        config = Config(ocr_language="eng")
        backend = TesseractBackend(config)
        assert backend._get_lang_code() == "eng"

    def test_returns_custom_language(self):
        config = Config(ocr_language="rus")
        backend = TesseractBackend(config)
        assert backend._get_lang_code() == "rus"


class TestOCREngineRecognize:
    """Tests for OCREngine.recognize() and _recognize_ensemble()."""

    def _engine_with_mocked_backend(self, text="hello", correction=False, engine="tesseract"):
        """Build an OCREngine whose Tesseract backend is fully mocked."""
        with (
            patch.object(TesseractBackend, "is_available", return_value=True),
            patch.object(TesseractBackend, "recognize", return_value=text),
        ):
            config = Config(
                ocr_engine=engine,
                enable_text_correction=correction,
                adaptive_ensemble=False,
            )
            return OCREngine(config)

    def test_single_backend_returns_recognized_text(self):
        engine = self._engine_with_mocked_backend("hello world")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(TesseractBackend, "recognize", return_value="hello world"):
            result = engine.recognize(arr)
        assert result == "hello world"

    def test_recognize_returns_empty_string_on_backend_exception(self):
        with (
            patch.object(TesseractBackend, "is_available", return_value=True),
            patch.object(TesseractBackend, "recognize", side_effect=RuntimeError("boom")),
        ):
            config = Config(ocr_engine="tesseract", enable_text_correction=False)
            engine = OCREngine(config)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(TesseractBackend, "recognize", side_effect=RuntimeError("boom")):
            result = engine.recognize(arr)
        assert result == ""

    def test_recognize_applies_text_correction(self):
        with patch.object(TesseractBackend, "is_available", return_value=True):
            config = Config(
                ocr_engine="tesseract",
                enable_text_correction=True,
                ocr_language="eng",
                adaptive_ensemble=False,
            )
            engine = OCREngine(config)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(TesseractBackend, "recognize", return_value="1 am happy"):
            result = engine.recognize(arr)
        assert "I am" in result

    def test_recognize_ensemble_no_available_backends_returns_empty(self):
        with patch.object(TesseractBackend, "is_available", return_value=True):
            config = Config(ocr_engine="tesseract", enable_text_correction=False)
            engine = OCREngine(config)
        # Mark all backends unavailable to simulate _recognize_ensemble with nothing
        for b in engine.backends.values():
            b._available = False
        result = engine._recognize_ensemble(Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)))
        assert result == ""

    def test_recognize_ensemble_single_result_passthrough(self):
        with patch.object(TesseractBackend, "is_available", return_value=True):
            config = Config(ocr_engine="tesseract", enable_text_correction=False)
            engine = OCREngine(config)
        with (
            patch.object(TesseractBackend, "is_available", return_value=True),
            patch.object(TesseractBackend, "recognize", return_value="only result"),
            patch.object(EasyOCRBackend, "is_available", return_value=False),
        ):
            result = engine._recognize_ensemble(
                Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
            )
        assert result == "only result"

    def test_recognize_ensemble_combines_two_backends(self):
        with patch.object(TesseractBackend, "is_available", return_value=True):
            config = Config(ocr_engine="tesseract", enable_text_correction=False)
            engine = OCREngine(config)
        with (
            patch.object(TesseractBackend, "is_available", return_value=True),
            patch.object(TesseractBackend, "recognize", return_value="hello world"),
            patch.object(EasyOCRBackend, "is_available", return_value=True),
            patch.object(EasyOCRBackend, "recognize", return_value="hello world"),
        ):
            result = engine._recognize_ensemble(
                Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
            )
        assert "hello" in result
        assert "world" in result
