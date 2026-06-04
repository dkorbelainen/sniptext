from sniptext.ocr import OCRBackend


def test_base_recognize_detailed_returns_none_conf():
    """Default implementation: text from recognize(), conf None."""

    class Dummy(OCRBackend):
        def recognize(self, image):
            return "hello world"

        def is_available(self):
            return True

    text, confs = Dummy().recognize_detailed(image=None)
    assert text == "hello world"
    assert confs is None


def test_easyocr_broadcast_groups_per_detection(monkeypatch):
    """EasyOCR detection confidence is broadcast to each word in the detection,
    one line per detection."""
    from sniptext.config import Config
    from sniptext.ocr import EasyOCRBackend

    backend = EasyOCRBackend(Config())

    class FakeReader:
        def readtext(self, *a, **k):
            return [(None, "hello world", 0.8), (None, "foo", 0.6)]

    backend._reader = FakeReader()
    backend._initialized = True
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(backend, "_lazy_init", lambda: None)
    backend.config.ocr_confidence_threshold = 0.0

    import numpy as np
    from PIL import Image

    text, confs = backend.recognize_detailed(Image.fromarray(np.zeros((4, 4), dtype="uint8")))
    assert text == "hello world\nfoo"
    assert confs == [[0.8, 0.8], [0.6]]
