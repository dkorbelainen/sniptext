"""SnipText - Lightweight OCR Screen Capture Utility."""

__version__ = "0.1.0"
__author__ = "dkorbelainen"
__license__ = "MIT"

from .capture import ScreenCapture
from .ocr import OCREngine
from .clipboard import ClipboardManager
from .config import Config

__all__ = [
    "ScreenCapture",
    "OCREngine",
    "ClipboardManager",
    "Config",
]
