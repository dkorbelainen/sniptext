"""
SnipText - Lightweight OCR Screen Capture Utility
Main entry point for the application.
"""

import argparse
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

from loguru import logger

__version__ = _pkg_version("sniptext")


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    if verbose:
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
    else:
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> {message}",
            level="INFO",
        )


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="SnipText - OCR Screen Capture Utility")
    parser.add_argument(
        "--version",
        action="version",
        version=f"sniptext {__version__}",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path.home() / ".config" / "sniptext" / "config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--capture-now",
        action="store_true",
        help="Capture screen immediately without hotkey",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["tesseract", "easyocr", "ensemble"],
        help="OCR engine to use (default: from config or tesseract)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available OCR backends",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current configuration and exit",
    )

    args = parser.parse_args()

    from sniptext.capture import ScreenCapture
    from sniptext.clipboard import ClipboardManager
    from sniptext.config import Config
    from sniptext.hotkey import HotkeyManager
    from sniptext.ocr import OCREngine

    setup_logging(args.verbose)
    logger.info("Starting SnipText...")

    config = Config.load(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    if args.list_models:
        ocr = OCREngine(config)
        print("Available OCR backends:")
        for name in ocr.get_available_backends():
            print(f"  • {name}")
        return 0

    if args.print_config:
        # Prefer a public render_config() method if available, fall back to the private helper for compatibility
        render = getattr(config, "render_config", None)
        if callable(render):
            print(render(), end="")
        else:
            private_render = getattr(config, "_render_config")
            print(private_render(), end="")
        return 0

    if args.ocr_engine:
        config.ocr_engine = args.ocr_engine
        logger.info(f"OCR engine overridden to: {args.ocr_engine}")

    hotkey_manager = None
    clipboard_manager = None
    try:
        screen_capture = ScreenCapture(config)
        ocr_engine = OCREngine(config)
        clipboard_manager = ClipboardManager()

        logger.info("Components initialized successfully")

        if args.capture_now:
            logger.info("Capturing screen...")
            image = screen_capture.capture_region()

            if image is not None:
                logger.info("Running OCR...")
                text = ocr_engine.recognize(image)

                if text:
                    copied = clipboard_manager.copy(text)
                    if copied:
                        print(f"✓ Copied {len(text)} characters to clipboard:\n")
                        print(text)
                    else:
                        logger.error("Failed to copy text to clipboard")
                        print(text)
                        return 1
                else:
                    print("✗ No text recognized in the selected area")
            else:
                logger.error("Failed to capture screen")
                return 1
        else:
            logger.info("Starting hotkey daemon...")
            hotkey_manager = HotkeyManager(
                config=config,
                screen_capture=screen_capture,
                ocr_engine=ocr_engine,
                clipboard_manager=clipboard_manager,
            )

            print(f"\nSnipText {__version__} is running")
            print(f"  Hotkey  : {config.hotkey}")
            print(f"  Engine  : {config.ocr_engine}")
            print(f"  Config  : {args.config}")
            print("\nPress Ctrl+C to quit\n")

            hotkey_manager.start()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if hotkey_manager is not None:
            hotkey_manager.stop()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception(e)
        return 1
    finally:
        if clipboard_manager is not None:
            clipboard_manager.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
