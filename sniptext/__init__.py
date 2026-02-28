"""SnipText - Lightweight OCR Screen Capture Utility."""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("sniptext")
except Exception:
    __version__ = "unknown"
__author__ = "dkorbelainen"
__license__ = "MIT"

from .capture import ScreenCapture
from .clipboard import ClipboardManager
from .config import Config
from .ocr import OCREngine

__all__ = [
    "ScreenCapture",
    "OCREngine",
    "ClipboardManager",
    "Config",
]
