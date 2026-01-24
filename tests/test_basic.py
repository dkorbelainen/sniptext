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
    from sniptext.analyzer import ImageAnalyzer
    from PIL import Image
    import numpy as np

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


