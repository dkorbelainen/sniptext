"""Tests for sniptext.__main__ CLI entry point."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from sniptext.__main__ import main, setup_logging

# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_verbose_level_is_debug(self):
        from loguru import logger

        with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
            setup_logging(verbose=True)
            _, kwargs = mock_add.call_args
            assert kwargs["level"] == "DEBUG"

    def test_default_level_is_info(self):
        from loguru import logger

        with patch.object(logger, "remove"), patch.object(logger, "add") as mock_add:
            setup_logging(verbose=False)
            _, kwargs = mock_add.call_args
            assert kwargs["level"] == "INFO"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _run_main(argv, config=None):
    """Patch sys.argv and common imports, then call main()."""
    if config is None:
        config = MagicMock()
        config._render_config.return_value = "hotkey: <ctrl>+<alt>+t\n"
        config.ocr_engine = "ensemble"
        config.notification_enabled = True

    with (
        patch("sys.argv", ["sniptext"] + argv),
        patch("sniptext.config.Config") as MockConfig,
        patch("sniptext.ocr.OCREngine") as MockOCR,
        patch("sniptext.capture.ScreenCapture") as MockCapture,
        patch("sniptext.clipboard.ClipboardManager") as MockClipboard,
        patch("sniptext.hotkey.HotkeyManager") as MockHotkey,
        patch("sniptext.__main__.setup_logging"),
    ):
        MockConfig.load.return_value = config
        yield MockConfig, MockOCR, MockCapture, MockClipboard, MockHotkey, config


# ---------------------------------------------------------------------------
# --print-config
# ---------------------------------------------------------------------------


class TestPrintConfig:
    def test_prints_config_and_exits_zero(self, capsys):
        with _run_main(["--print-config"]) as (_, __, ___, ____, _____, config):
            result = main()

        assert result == 0
        assert "hotkey" in capsys.readouterr().out

    def test_does_not_start_capture_components(self):
        with _run_main(["--print-config"]) as (_, MockOCR, MockCapture, MockClipboard, _, __):
            main()

        MockOCR.assert_not_called()
        MockCapture.assert_not_called()
        MockClipboard.assert_not_called()


# ---------------------------------------------------------------------------
# --list-models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_lists_backends_and_exits_zero(self, capsys):
        with _run_main(["--list-models"]) as (_, MockOCR, ___, ____, _____, config):
            MockOCR.return_value.get_available_backends.return_value = ["tesseract", "ensemble"]
            result = main()

        assert result == 0
        out = capsys.readouterr().out
        assert "tesseract" in out
        assert "ensemble" in out

    def test_does_not_start_capture(self):
        with _run_main(["--list-models"]) as (_, MockOCR, MockCapture, MockClipboard, _, __):
            MockOCR.return_value.get_available_backends.return_value = []
            main()

        MockCapture.assert_not_called()
        MockClipboard.assert_not_called()


# ---------------------------------------------------------------------------
# --capture-now
# ---------------------------------------------------------------------------


class TestCaptureNow:
    def test_success_prints_text_returns_zero(self, capsys):
        fake_image = MagicMock()
        with _run_main(["--capture-now"]) as (_, MockOCR, MockCapture, MockClipboard, _, config):
            MockCapture.return_value.capture_region.return_value = fake_image
            MockOCR.return_value.recognize.return_value = "hello world"
            MockClipboard.return_value.copy.return_value = True

            result = main()

        assert result == 0
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_clipboard_failure_returns_nonzero(self):
        fake_image = MagicMock()
        with _run_main(["--capture-now"]) as (_, MockOCR, MockCapture, MockClipboard, _, __):
            MockCapture.return_value.capture_region.return_value = fake_image
            MockOCR.return_value.recognize.return_value = "hello"
            MockClipboard.return_value.copy.return_value = False

            result = main()

        assert result == 1

    def test_no_text_recognized_prints_message(self, capsys):
        fake_image = MagicMock()
        with _run_main(["--capture-now"]) as (_, MockOCR, MockCapture, MockClipboard, _, __):
            MockCapture.return_value.capture_region.return_value = fake_image
            MockOCR.return_value.recognize.return_value = ""
            result = main()

        assert result == 0
        assert "No text" in capsys.readouterr().out

    def test_capture_returns_none_exits_one(self):
        with _run_main(["--capture-now"]) as (_, MockOCR, MockCapture, MockClipboard, _, __):
            MockCapture.return_value.capture_region.return_value = None
            result = main()

        assert result == 1

    def test_exception_returns_one(self):
        with _run_main(["--capture-now"]) as (_, MockOCR, MockCapture, _, __, ___):
            MockCapture.return_value.capture_region.side_effect = RuntimeError("boom")
            result = main()

        assert result == 1

    def test_keyboard_interrupt_returns_zero(self):
        with _run_main(["--capture-now"]) as (_, MockOCR, MockCapture, _, __, ___):
            MockCapture.return_value.capture_region.side_effect = KeyboardInterrupt
            result = main()

        assert result == 0


# ---------------------------------------------------------------------------
# --ocr-engine override
# ---------------------------------------------------------------------------


class TestOcrEngineOverride:
    def test_override_sets_engine_on_config(self):
        with _run_main(["--capture-now", "--ocr-engine", "tesseract"]) as (
            _,
            MockOCR,
            MockCapture,
            MockClipboard,
            __,
            config,
        ):
            MockCapture.return_value.capture_region.return_value = None
            main()

        assert config.ocr_engine == "tesseract"


# ---------------------------------------------------------------------------
# sniptext.__init__ lazy imports
# ---------------------------------------------------------------------------


class TestInitLazyImports:
    def test_unknown_attribute_raises(self):
        import sniptext

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = sniptext.NonExistentThing


# ---------------------------------------------------------------------------
# --file IMAGE
# ---------------------------------------------------------------------------


class TestFileInput:
    def test_file_runs_ocr_and_copies(self, tmp_path, capsys):
        img_path = tmp_path / "test.png"
        fake_array = MagicMock()

        with (
            _run_main(["--file", str(img_path)]) as (_, MockOCR, ___, MockClipboard, __, ____),
            patch("PIL.Image.open") as MockOpen,
            patch("numpy.array", return_value=fake_array),
        ):
            MockOpen.return_value = MagicMock()
            MockOCR.return_value.recognize.return_value = "hello from file"
            MockClipboard.return_value.copy.return_value = True

            result = main()

        assert result == 0
        assert "hello from file" in capsys.readouterr().out

    def test_file_no_text_recognized(self, tmp_path, capsys):
        img_path = tmp_path / "blank.png"
        with (
            _run_main(["--file", str(img_path)]) as (_, MockOCR, ___, MockClipboard, __, ____),
            patch("PIL.Image.open") as MockOpen,
            patch("numpy.array", return_value=MagicMock()),
        ):
            MockOpen.return_value = MagicMock()
            MockOCR.return_value.recognize.return_value = ""
            MockClipboard.return_value.copy.return_value = True

            result = main()

        assert result == 0
        assert "No text" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --output FILE
# ---------------------------------------------------------------------------


class TestOutputFile:
    def test_output_writes_text_to_file(self, tmp_path, capsys):
        out_path = tmp_path / "result.txt"
        fake_image = MagicMock()
        with _run_main(["--capture-now", "--output", str(out_path)]) as (
            _,
            MockOCR,
            MockCapture,
            MockClipboard,
            __,
            ___,
        ):
            MockCapture.return_value.capture_region.return_value = fake_image
            MockOCR.return_value.recognize.return_value = "written text"
            MockClipboard.return_value.copy.return_value = True

            result = main()

        assert result == 0
        assert out_path.read_text() == "written text"

    def test_output_write_error_returns_one(self, tmp_path, capsys):
        fake_image = MagicMock()
        out_path = tmp_path / "missing" / "out.txt"  # parent dir not created
        with _run_main(["--capture-now", "--output", str(out_path)]) as (
            _,
            MockOCR,
            MockCapture,
            MockClipboard,
            __,
            ___,
        ):
            MockCapture.return_value.capture_region.return_value = fake_image
            MockOCR.return_value.recognize.return_value = "some text"
            MockClipboard.return_value.copy.return_value = True

            result = main()

        assert result == 1
