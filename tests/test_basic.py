"""Basic tests for SnipText."""

import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_defaults():
    """Test config has correct defaults."""
    from sniptext.config import Config

    config = Config()
    assert config.hotkey == "<ctrl>+<alt>+t"
    assert config.ocr_language == "eng"
    assert config.ocr_engine == "ensemble"


def test_analyzer_features():
    """Test image analyzer extracts features."""
    import numpy as np
    from PIL import Image

    from sniptext.analyzer import ImageAnalyzer

    analyzer = ImageAnalyzer()

    # Create test image
    img = Image.fromarray(np.full((100, 300, 3), 150, dtype=np.uint8))

    # Extract features
    features = analyzer.extract_features(img)

    assert len(features) == 5
    assert 0 <= features[0] <= 1  # brightness normalized


def test_imports():
    """Test that core modules can be imported."""
    from sniptext import Config, OCREngine
    from sniptext.analyzer import ImageAnalyzer

    assert Config is not None
    assert OCREngine is not None
    assert ImageAnalyzer is not None


def test_no_eager_heavy_imports(monkeypatch):
    """Regression: `import sniptext` must not load numpy/Pillow/pytesseract."""
    import importlib
    import sys

    heavy = {"numpy", "PIL", "pytesseract", "easyocr", "sklearn"}

    sniptext_keys = [k for k in sys.modules if k == "sniptext" or k.startswith("sniptext.")]
    for k in sniptext_keys:
        monkeypatch.delitem(sys.modules, k)

    before = set(sys.modules)
    importlib.import_module("sniptext")
    after = set(sys.modules)

    new_modules = after - before
    loaded_heavy = {m for m in new_modules for h in heavy if m == h or m.startswith(h + ".")}
    assert not loaded_heavy, f"Eager import of heavy deps detected: {loaded_heavy}"
