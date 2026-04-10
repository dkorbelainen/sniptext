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
        assert daemon.ocr_engine is None
        assert daemon.clipboard_manager is None

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

    @patch("sniptext.daemon.HTTPServer")
    @patch("sniptext.daemon.OCREngine")
    @patch("sniptext.daemon.ClipboardManager")
    @patch("sniptext.daemon.ScreenCapture")
    @patch("sniptext.daemon.HistoryManager")
    def test_daemon_component_initialization(
        self, mock_history, mock_capture, mock_clipboard, mock_ocr, mock_server, tmp_path
    ):
        """Test that daemon properly initializes all components and stores them."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "ocr_engine: tesseract\n"
            "display_server: x11\n"
            "hotkey: ctrl+alt+s\n"
            "history_enabled: true\n"
            "history_size: 50\n"
        )

        config = Config.load(config_file)

        # Mock HTTPServer to prevent actual server startup
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance

        # Mock serve_forever to prevent blocking
        mock_server_instance.serve_forever = MagicMock()

        # Create mocked components
        mock_ocr_instance = MagicMock()
        mock_clipboard_instance = MagicMock()
        mock_capture_instance = MagicMock()
        mock_history_instance = MagicMock()

        mock_ocr.return_value = mock_ocr_instance
        mock_clipboard.return_value = mock_clipboard_instance
        mock_capture.return_value = mock_capture_instance
        mock_history.return_value = mock_history_instance

        daemon = SnipTextDaemon(config, port=9879)

        # Start daemon (will initialize components)
        try:
            daemon.start()
        except Exception:
            pass  # Ignore any errors from mocked components

        # Verify components were initialized and stored
        assert daemon.ocr_engine is mock_ocr_instance
        assert daemon.clipboard_manager is mock_clipboard_instance
        assert daemon.screen_capture is mock_capture_instance
        assert daemon.history_manager is mock_history_instance

    @patch("sniptext.daemon.HTTPServer")
    @patch("sniptext.daemon.OCREngine")
    @patch("sniptext.daemon.ClipboardManager")
    @patch("sniptext.daemon.ScreenCapture")
    @patch("sniptext.daemon.HistoryManager")
    def test_daemon_disabled_history(
        self, mock_history, mock_capture, mock_clipboard, mock_ocr, mock_server, tmp_path
    ):
        """Test daemon with history disabled doesn't initialize HistoryManager."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "ocr_engine: tesseract\n"
            "display_server: x11\n"
            "hotkey: ctrl+alt+s\n"
            "history_enabled: false\n"
        )

        config = Config.load(config_file)

        # Mock HTTPServer to prevent actual server startup
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_server_instance.serve_forever = MagicMock()

        # Mock other components
        mock_ocr.return_value = MagicMock()
        mock_clipboard.return_value = MagicMock()
        mock_capture.return_value = MagicMock()

        daemon = SnipTextDaemon(config, port=9880)

        # Start daemon
        try:
            daemon.start()
        except Exception:
            pass

        # Verify HistoryManager was NOT initialized
        mock_history.assert_not_called()
        assert daemon.history_manager is None

    @patch("sniptext.daemon.HTTPServer")
    @patch("sniptext.daemon.OCREngine")
    @patch("sniptext.daemon.ClipboardManager")
    @patch("sniptext.daemon.ScreenCapture")
    def test_daemon_stop_cleanup(
        self, mock_capture, mock_clipboard, mock_ocr, mock_server, tmp_path
    ):
        """Test that daemon.stop() properly cleans up resources."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        config = Config.load(config_file)

        # Mock HTTPServer
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_server_instance.serve_forever = MagicMock()

        # Mock components
        mock_ocr_instance = MagicMock()
        mock_clipboard_instance = MagicMock()
        mock_capture_instance = MagicMock()

        mock_ocr.return_value = mock_ocr_instance
        mock_clipboard.return_value = mock_clipboard_instance
        mock_capture.return_value = mock_capture_instance

        daemon = SnipTextDaemon(config, port=9881)

        # Initialize components
        try:
            daemon.start()
        except Exception:
            pass

        # Call stop
        daemon.stop()

        # Verify cleanup was called
        mock_server_instance.shutdown.assert_called_once()
        mock_server_instance.server_close.assert_called_once()
        mock_clipboard_instance.cleanup.assert_called_once()
