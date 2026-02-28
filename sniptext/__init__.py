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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ScreenCapture": ("sniptext.capture", "ScreenCapture"),
    "ClipboardManager": ("sniptext.clipboard", "ClipboardManager"),
    "Config": ("sniptext.config", "Config"),
    "OCREngine": ("sniptext.ocr", "OCREngine"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'sniptext' has no attribute {name!r}")
    import importlib
    import sys

    module_name, attr = _LAZY_IMPORTS[name]
    obj = getattr(importlib.import_module(module_name), attr)
    setattr(sys.modules[__name__], name, obj)
    return obj
