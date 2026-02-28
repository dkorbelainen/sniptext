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
    import sys

    _public = {
        "ScreenCapture": ("sniptext.capture", "ScreenCapture"),
        "ClipboardManager": ("sniptext.clipboard", "ClipboardManager"),
        "Config": ("sniptext.config", "Config"),
        "OCREngine": ("sniptext.ocr", "OCREngine"),
    }
    if name not in _public:
        raise AttributeError(f"module 'sniptext' has no attribute {name!r}")
    module_name, attr = _public[name]
    import importlib

    obj = getattr(importlib.import_module(module_name), attr)
    # Cache in module namespace so subsequent accesses skip __getattr__
    setattr(sys.modules[__name__], name, obj)
    return obj
