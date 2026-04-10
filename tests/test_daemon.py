"""Tests for sniptext.daemon HTTP server."""

from unittest.mock import MagicMock, patch

from sniptext.config import Config
from sniptext.daemon import SnipTextDaemon


class TestSnipTextDaemon:
    """Tests for daemon initialization and configuration."""

    def test_daemon_init(self, tmp_path):
        """Test daemon initialization."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        config = Config.load(config_file)
        daemon = SnipTextDaemon(config, port=9877)

        assert daemon.port == 9877
        assert daemon.config == config
        assert daemon.server is None
        assert daemon.thread is None

    def test_daemon_with_history_enabled(self, tmp_path):
        """Test daemon initialization with history enabled."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "ocr_engine: tesseract\n"
            "display_server: x11\n"
            "hotkey: ctrl+alt+s\n"
            "history_enabled: true\n"
            "history_size: 50\n"
        )

        config = Config.load(config_file)
        daemon = SnipTextDaemon(config, port=9878)

        assert daemon.config.history_enabled is True
        assert daemon.config.history_size == 50

    @patch("sniptext.daemon.OCREngine")
    @patch("sniptext.daemon.ClipboardManager")
    @patch("sniptext.daemon.ScreenCapture")
    @patch("sniptext.daemon.HistoryManager")
    def test_daemon_component_initialization(
        self, mock_history, mock_capture, mock_clipboard, mock_ocr, tmp_path
    ):
        """Test that daemon properly initializes all components."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "ocr_engine: tesseract\n"
            "display_server: x11\n"
            "hotkey: ctrl+alt+s\n"
            "history_enabled: true\n"
            "history_size: 50\n"
        )

        config = Config.load(config_file)

        # Mock the components to prevent actual initialization
        mock_ocr.return_value = MagicMock()
        mock_clipboard.return_value = MagicMock()
        mock_capture.return_value = MagicMock()
        mock_history.return_value = MagicMock()

        # We can't fully test start() in unit tests without running a real server,
        # but we can verify the daemon is properly configured
        daemon = SnipTextDaemon(config, port=9879)
        assert daemon.config.history_enabled is True

    @patch("sniptext.daemon.OCREngine")
    @patch("sniptext.daemon.ClipboardManager")
    @patch("sniptext.daemon.ScreenCapture")
    def test_daemon_disabled_history(self, mock_capture, mock_clipboard, mock_ocr, tmp_path):
        """Test daemon with history disabled."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "ocr_engine: tesseract\n"
            "display_server: x11\n"
            "hotkey: ctrl+alt+s\n"
            "history_enabled: false\n"
        )

        config = Config.load(config_file)
        daemon = SnipTextDaemon(config, port=9880)

        assert daemon.config.history_enabled is False
