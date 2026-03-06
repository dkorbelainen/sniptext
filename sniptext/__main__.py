"""
SnipText - Lightweight OCR Screen Capture Utility
Main entry point for the application.
"""

import argparse
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Optional

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


def _output_result(
    text: str,
    clipboard_manager,
    output_path: Optional[Path],
    history_manager=None,
) -> int:
    """Print, copy, and optionally write OCR result. Returns exit code."""
    if not text:
        print("✗ No text recognized")
        return 0

    print(text)

    copied = clipboard_manager.copy(text)
    if copied:
        print(f"\n✓ Copied {len(text)} characters to clipboard")
    else:
        logger.error("Failed to copy text to clipboard")

    if output_path is not None:
        try:
            output_path.write_text(text, encoding="utf-8")
            print(f"✓ Saved to {output_path}")
        except OSError as e:
            logger.error(f"Failed to write output file: {e}")
            return 1

    if history_manager is not None:
        history_manager.append(text)

    return 0 if copied else 1


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
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--capture-now",
        action="store_true",
        help="Capture screen immediately without hotkey",
    )
    capture_group.add_argument(
        "--file",
        type=Path,
        metavar="IMAGE",
        help="Run OCR on an image file instead of capturing the screen",
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
    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Write recognized text to FILE (in addition to clipboard). Only valid with --file or --capture-now.",
    )
    parser.add_argument(
        "--history",
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Print last N captured texts (default 10) and exit",
    )
    parser.add_argument(
        "--profile",
        type=str,
        metavar="NAME",
        help="Apply a named config profile from ~/.config/sniptext/profiles/NAME.yaml",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available config profiles and exit",
    )

    args = parser.parse_args()

    if args.output and not (args.file or args.capture_now):
        parser.error(
            "--output requires either --file or --capture-now; it is not supported in hotkey-only mode."
        )

    from sniptext.config import Config
    from sniptext.history import HistoryManager

    setup_logging(args.verbose)
    logger.info("Starting SnipText...")

    if args.list_profiles:
        profiles = Config.list_profiles(args.config)
        if not profiles:
            print("No profiles found.")
            print(f"  Create YAML files in: {args.config.parent / 'profiles'}/")
        else:
            print("Available profiles:")
            for name in profiles:
                print(f"  • {name}")
        return 0

    if args.profile:
        try:
            config = Config.load_with_profile(args.config, args.profile)
        except FileNotFoundError as e:
            print(f"✗ {e}")
            print("  Run 'sniptext --list-profiles' to see available profiles.")
            return 1
        logger.info(f"Loaded config with profile {args.profile!r}")
    else:
        config = Config.load(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    if args.history is not None:
        history_manager = HistoryManager(max_size=config.history_size)
        entries = history_manager.read(args.history)
        if not entries:
            print("No history yet.")
        else:
            for entry in entries:
                timestamp = entry.get("timestamp")
                text = entry.get("text")
                if timestamp is None or text is None:
                    logger.warning(
                        "Skipping invalid history entry without required fields: {}", entry
                    )
                    continue
                print(f"[{timestamp}]")
                print(text)
                print()
        return 0

    from sniptext.capture import ScreenCapture
    from sniptext.clipboard import ClipboardManager
    from sniptext.hotkey import HotkeyManager
    from sniptext.ocr import OCREngine

    if args.list_models:
        ocr = OCREngine(config)
        print("Available OCR backends:")
        for name in ocr.get_available_backends():
            print(f"  • {name}")
        return 0

    if args.print_config:
        print(config._render_config(), end="")
        return 0

    if args.ocr_engine:
        config.ocr_engine = args.ocr_engine
        logger.info(f"OCR engine overridden to: {args.ocr_engine}")

    hotkey_manager = None
    clipboard_manager = None
    history_manager = (
        HistoryManager(max_size=config.history_size) if config.history_enabled else None
    )
    try:
        ocr_engine = OCREngine(config)
        clipboard_manager = ClipboardManager()

        logger.info("Components initialized successfully")

        if args.file:
            logger.info(f"Loading image from {args.file}...")
            import numpy as np
            from PIL import Image as _PIL_Image
            from PIL import UnidentifiedImageError as _PIL_UnidentifiedImageError

            try:
                with _PIL_Image.open(args.file) as pil_image:
                    image = np.array(pil_image)
            except (OSError, _PIL_UnidentifiedImageError) as e:
                logger.error(f"Failed to open image file '{args.file}': {e}")
                return 2
        elif args.capture_now:
            screen_capture = ScreenCapture(config)
            logger.info("Capturing screen...")
            image = screen_capture.capture_region()
            if image is None:
                logger.error("Failed to capture screen")
                return 1
        else:
            image = None

        if image is not None:
            logger.info("Running OCR...")
            text = ocr_engine.recognize(image)
            rc = _output_result(text, clipboard_manager, args.output, history_manager)
            if rc != 0:
                return rc
        else:
            screen_capture = ScreenCapture(config)
            logger.info("Starting hotkey daemon...")
            hotkey_manager = HotkeyManager(
                config=config,
                screen_capture=screen_capture,
                ocr_engine=ocr_engine,
                clipboard_manager=clipboard_manager,
                history_manager=history_manager,
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
