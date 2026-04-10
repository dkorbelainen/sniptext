"""Daemon server for sniptext with REST API."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Optional

from loguru import logger

from .capture import ScreenCapture
from .clipboard import ClipboardManager
from .config import Config
from .history import HistoryManager
from .ocr import OCREngine


class SnipTextHandler(BaseHTTPRequestHandler):
    """HTTP request handler for SnipText daemon."""

    # Class-level references shared across all handlers
    ocr_engine: Optional[OCREngine] = None
    clipboard_manager: Optional[ClipboardManager] = None
    screen_capture: Optional[ScreenCapture] = None
    history_manager: Optional[HistoryManager] = None
    config: Optional[Config] = None

    def log_message(self, format, *args):
        """Suppress default HTTP logging in favour of loguru."""
        logger.debug(f"HTTP {format % args}")

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/capture":
            self._handle_capture()
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith("/history"):
            self._handle_history()
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def _handle_capture(self):
        """Handle POST /capture: capture screen and run OCR."""
        if self.screen_capture is None or self.ocr_engine is None:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Engine not initialized"}).encode())
            return

        try:
            logger.info("Capture requested via daemon API")
            image = self.screen_capture.capture_region()
            if image is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Capture cancelled or failed"}).encode())
                return

            text = self.ocr_engine.recognize(image)
            if not text:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"text": "", "error": "No text recognized"}).encode())
                return

            # Copy to clipboard
            if self.clipboard_manager:
                self.clipboard_manager.copy(text)

            # Save to history
            if self.history_manager:
                self.history_manager.append(text)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"text": text, "length": len(text)}).encode())
            logger.info(f"Captured {len(text)} chars via daemon API")

        except Exception as e:
            logger.error(f"Capture error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_history(self):
        """Handle GET /history[?n=N]: retrieve history."""
        if self.history_manager is None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"entries": []}).encode())
            return

        try:
            n = 10  # default
            if "?" in self.path:
                query = self.path.split("?")[1]
                for param in query.split("&"):
                    if param.startswith("n="):
                        try:
                            n = int(param[2:])
                        except ValueError:
                            pass

            entries = self.history_manager.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"entries": entries}).encode())

        except Exception as e:
            logger.error(f"History error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


class SnipTextDaemon:
    """Daemon server for SnipText with REST API."""

    def __init__(self, config: Config, port: int = 9877):
        """Initialize daemon."""
        self.config = config
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[Thread] = None

    def start(self):
        """Start the daemon server."""
        try:
            logger.info(f"Initializing SnipText daemon on port {self.port}...")

            # Initialize components
            ocr_engine = OCREngine(self.config)
            clipboard_manager = ClipboardManager()
            screen_capture = ScreenCapture(self.config)
            history_manager = (
                HistoryManager(max_size=self.config.history_size)
                if self.config.history_enabled
                else None
            )

            # Set class-level references for handler
            SnipTextHandler.ocr_engine = ocr_engine
            SnipTextHandler.clipboard_manager = clipboard_manager
            SnipTextHandler.screen_capture = screen_capture
            SnipTextHandler.history_manager = history_manager
            SnipTextHandler.config = self.config

            # Create server
            self.server = HTTPServer(("localhost", self.port), SnipTextHandler)
            logger.info(f"SnipText daemon listening on http://localhost:{self.port}")
            logger.info("Endpoints: POST /capture, GET /history?n=N, GET /health")
            logger.info("Press Ctrl+C to stop")

            # Run in main thread (blocking)
            self.server.serve_forever()

        except KeyboardInterrupt:
            logger.info("Daemon shutdown requested")
            self.stop()
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            raise

    def stop(self):
        """Stop the daemon server."""
        if self.server:
            self.server.shutdown()
            logger.info("Daemon stopped")
