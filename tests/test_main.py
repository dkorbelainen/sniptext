"""Tests for sniptext.__main__ CLI entry point."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

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
        config._render_config.return_value = "ocr_language: eng\n"
        config.notification_enabled = True
        config.history_enabled = False
        config.history_size = 50

    with (
        patch("sys.argv", ["sniptext"] + argv),
        patch("sniptext.config.Config") as MockConfig,
        patch("sniptext.ocr.OCREngine") as MockOCR,
        patch("sniptext.capture.ScreenCapture") as MockCapture,
        patch("sniptext.clipboard.ClipboardManager") as MockClipboard,
        patch("sniptext.notify.send") as MockSend,
        patch("sniptext.__main__.setup_logging"),
        patch("sniptext.history.HistoryManager"),
    ):
        MockConfig.load.return_value = config
        yield MockConfig, MockOCR, MockCapture, MockClipboard, MockSend, config


# ---------------------------------------------------------------------------
# --print-config
# ---------------------------------------------------------------------------


class TestPrintConfig:
    def test_prints_config_and_exits_zero(self, capsys):
        with _run_main(["--print-config"]) as (_, __, ___, ____, _____, config):
            result = main()

        assert result == 0
        assert "ocr_language" in capsys.readouterr().out

    def test_does_not_start_capture_components(self):
        with _run_main(["--print-config"]) as (_, MockOCR, MockCapture, MockClipboard, _, __):
            main()

        MockOCR.assert_not_called()
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

        with (
            _run_main(["--file", str(img_path)]) as (_, MockOCR, ___, MockClipboard, __, ____),
            patch("PIL.Image.open") as MockOpen,
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


# ---------------------------------------------------------------------------
# --history
# ---------------------------------------------------------------------------


class TestHistory:
    def test_history_empty_prints_message(self, capsys):
        with (
            patch("sys.argv", ["sniptext", "--history"]),
            patch("sniptext.config.Config") as MockConfig,
            patch("sniptext.__main__.setup_logging"),
            patch("sniptext.history.HistoryManager") as MockHM,
        ):
            cfg = MagicMock()
            cfg.history_size = 50
            MockConfig.load.return_value = cfg
            MockHM.return_value.read.return_value = []

            result = main()

        assert result == 0
        assert "No history" in capsys.readouterr().out

    def test_history_prints_entries(self, capsys):
        with (
            patch("sys.argv", ["sniptext", "--history", "5"]),
            patch("sniptext.config.Config") as MockConfig,
            patch("sniptext.__main__.setup_logging"),
            patch("sniptext.history.HistoryManager") as MockHM,
        ):
            cfg = MagicMock()
            cfg.history_size = 50
            MockConfig.load.return_value = cfg
            MockHM.return_value.read.return_value = [
                {"timestamp": "2024-01-01T00:00:00+00:00", "text": "captured text"}
            ]

            result = main()

        assert result == 0
        out = capsys.readouterr().out
        assert "captured text" in out
        assert "2024-01-01" in out


# ---------------------------------------------------------------------------
# --profile / --list-profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_list_profiles_no_profiles(self, capsys):
        with (
            patch("sys.argv", ["sniptext", "--list-profiles"]),
            patch("sniptext.config.Config") as MockConfig,
            patch("sniptext.__main__.setup_logging"),
        ):
            MockConfig.list_profiles.return_value = []
            result = main()

        assert result == 0
        assert "No profiles" in capsys.readouterr().out

    def test_list_profiles_shows_names(self, capsys):
        with (
            patch("sys.argv", ["sniptext", "--list-profiles"]),
            patch("sniptext.config.Config") as MockConfig,
            patch("sniptext.__main__.setup_logging"),
        ):
            MockConfig.list_profiles.return_value = ["fast", "gpu"]
            result = main()

        assert result == 0
        out = capsys.readouterr().out
        assert "fast" in out
        assert "gpu" in out

    def test_a_missing_profile_is_a_bad_argument(self, capsys):
        with (
            patch("sys.argv", ["sniptext", "--profile", "nosuch", "--capture-now"]),
            patch("sniptext.config.Config") as MockConfig,
            patch("sniptext.__main__.setup_logging"),
        ):
            MockConfig.load_with_profile.side_effect = FileNotFoundError(
                "Profile 'nosuch' not found"
            )
            result = main()

        assert result == 2
        assert "nosuch" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# default action, notifications, failures
# ---------------------------------------------------------------------------


class TestSingleCapture:
    def test_no_arguments_captures_once(self, capsys):
        with _run_main([]) as (_, MockOCR, MockCapture, MockClipboard, _send, _config):
            MockCapture.return_value.capture_region.return_value = np.zeros((4, 4, 3), np.uint8)
            MockOCR.return_value.recognize.return_value = "hello"
            MockClipboard.return_value.copy.return_value = True
            assert main() == 0
        assert MockCapture.return_value.capture_region.call_count == 1
        assert "hello" in capsys.readouterr().out

    @pytest.mark.parametrize("flag", ["--client", "--interactive", "--list-models", "--benchmark"])
    def test_removed_flags_are_rejected(self, flag):
        with _run_main([flag]), pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2

    def test_serve_command_is_rejected(self):
        with _run_main(["serve"]), pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2


class TestNotifications:
    def capture(self, text, argv=()):
        with _run_main(list(argv)) as (_, MockOCR, MockCapture, MockClipboard, MockSend, _config):
            MockCapture.return_value.capture_region.return_value = np.zeros((4, 4, 3), np.uint8)
            MockOCR.return_value.recognize.return_value = text
            MockClipboard.return_value.copy.return_value = True
            code = main()
        return code, MockSend

    def test_a_capture_notifies_with_a_preview(self):
        code, send = self.capture("hello world")
        assert code == 0 and send.call_args.args == ("✓ hello world",)

    def test_an_empty_capture_says_so(self):
        _, send = self.capture("")
        assert send.call_args.args == ("No text found in selected area",)

    def test_a_failed_copy_is_notified(self):
        with _run_main([]) as (_, MockOCR, MockCapture, MockClipboard, MockSend, _config):
            MockCapture.return_value.capture_region.return_value = np.zeros((4, 4, 3), np.uint8)
            MockOCR.return_value.recognize.return_value = "hello"
            MockClipboard.return_value.copy.return_value = False
            assert main() == 1
        assert MockSend.call_args.args == ("✗ Could not copy the text to the clipboard",)

    def test_no_notification_when_disabled(self):
        config = MagicMock(notification_enabled=False, history_enabled=False, history_size=50)
        with _run_main([], config) as (_, MockOCR, MockCapture, MockClipboard, MockSend, _c):
            MockCapture.return_value.capture_region.return_value = np.zeros((4, 4, 3), np.uint8)
            MockOCR.return_value.recognize.return_value = "hello"
            MockClipboard.return_value.copy.return_value = True
            main()
        assert MockSend.call_count == 0

    def test_a_file_run_does_not_notify(self, tmp_path):
        path = tmp_path / "a.png"
        Image.new("RGB", (8, 8), "white").save(path)
        _, send = self.capture("hello", ["--file", str(path)])
        assert send.call_count == 0


class TestOcrFailure:
    def test_the_reason_is_shown_and_the_exit_code_is_one(self):
        from sniptext.ocr import OCRError

        with _run_main([]) as (_, MockOCR, MockCapture, _clip, MockSend, _config):
            MockCapture.return_value.capture_region.return_value = np.zeros((4, 4, 3), np.uint8)
            MockOCR.return_value.recognize.side_effect = OCRError("Tesseract failed: no 'ell'")
            assert main() == 1
        assert "no 'ell'" in MockSend.call_args.args[0]

    def test_a_palette_png_reaches_the_engine_as_an_image(self, tmp_path):
        path = tmp_path / "p.png"
        Image.new("RGB", (8, 8), (200, 30, 30)).convert("P").save(path)
        with _run_main(["--file", str(path)]) as (_, MockOCR, _cap, MockClipboard, _send, _c):
            MockOCR.return_value.recognize.return_value = "x"
            MockClipboard.return_value.copy.return_value = True
            assert main() == 0
        argument = MockOCR.return_value.recognize.call_args.args[0]
        assert isinstance(argument, Image.Image) and argument.mode == "P"
