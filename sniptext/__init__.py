"""SnipText - Lightweight OCR Screen Capture Utility."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("sniptext")
except PackageNotFoundError:
    __version__ = "unknown"
__author__ = "dkorbelainen"
__license__ = "MIT"

__all__ = [
    "ClipboardManager",
    "Config",
    "OCREngine",
    "ScreenCapture",
]


def __getattr__(name: str):
    if name == "ScreenCapture":
        from .capture import ScreenCapture

        return ScreenCapture
    if name == "ClipboardManager":
        from .clipboard import ClipboardManager

        return ClipboardManager
    if name == "Config":
        from .config import Config

        return Config
    if name == "OCREngine":
        from .ocr import OCREngine

        return OCREngine
    raise AttributeError(f"module 'sniptext' has no attribute {name!r}")
