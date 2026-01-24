"""Basic tests for SnipText."""

import pytest
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    try:
        from sniptext import Config, ScreenCapture, OCREngine, ClipboardManager
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_config_creation():
    """Test config creation with defaults."""
    from sniptext.config import Config

    config = Config()
    assert config.hotkey == "<ctrl>+<shift>+s"
    assert config.ocr_language == "en"
    assert config.num_threads == 4


def test_config_save_load(tmp_path):
    """Test config save and load."""
    from sniptext.config import Config

    config_path = tmp_path / "config.yaml"

    # Create and save config
    config1 = Config(hotkey="<ctrl>+s", ocr_language="ru")
    config1.save(config_path)

    # Load config
    config2 = Config.load(config_path)

    assert config2.hotkey == "<ctrl>+s"
    assert config2.ocr_language == "ru"


def test_clipboard_detection():
    """Test clipboard tool detection."""
    from sniptext.clipboard import ClipboardManager

    try:
        clipboard = ClipboardManager()
        assert clipboard.tool in ("wayland", "x11")
    except RuntimeError:
        # No clipboard tool available - that's ok for CI
        pytest.skip("No clipboard tool available")

