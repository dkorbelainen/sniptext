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
    notify: bool = False,
) -> int:
    """Print, copy, optionally write and record the OCR result. Returns the exit code."""
    from sniptext.notify import preview, send

    if not text:
        print("✗ No text recognized")
        if notify:
            send("No text found in selected area")
        return 0

    print(text)
    if not clipboard_manager.copy(text):
        logger.error("Failed to copy text to clipboard")
        return 1
    print(f"\n✓ Copied {len(text)} characters to clipboard")

    if output_path is not None:
        try:
            output_path.write_text(text, encoding="utf-8")
            print(f"✓ Saved to {output_path}")
        except OSError as e:
            logger.error(f"Failed to write output file: {e}")
            return 1

    if history_manager is not None:
        history_manager.append(text)
    if notify:
        send(f"✓ {preview(text)}")
    return 0


def _print_history(history_manager, count: int) -> None:
    entries = history_manager.read(count)
    if not entries:
        print("No history yet.")
        return
    for entry in entries:
        timestamp = entry.get("timestamp")
        text = entry.get("text")
        if timestamp is None or text is None:
            logger.warning("Skipping invalid history entry without required fields: {}", entry)
            continue
        print(f"[{timestamp}]")
        print(text)
        print()


def main():
    """Capture a screen region (or read an image file), recognise its text, copy it."""
    parser = argparse.ArgumentParser(description="SnipText - OCR Screen Capture Utility")
    parser.add_argument("--version", action="version", version=f"sniptext {__version__}")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path.home() / ".config" / "sniptext" / "config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--capture-now",
        action="store_true",
        help="Select a screen region and recognise it (the default action)",
    )
    source.add_argument(
        "--file",
        type=Path,
        metavar="IMAGE",
        help="Run OCR on an image file instead of capturing the screen",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print current configuration and exit"
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Write recognized text to FILE (in addition to clipboard)",
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
        help=(
            "Apply a named config profile from PROFILES_DIR/NAME.yaml, where PROFILES_DIR is the "
            "'profiles' directory next to the config file (default: ~/.config/sniptext/profiles)"
        ),
    )
    parser.add_argument(
        "--list-profiles", action="store_true", help="List available config profiles and exit"
    )
    args = parser.parse_args()

    from sniptext.config import Config
    from sniptext.history import HistoryManager

    setup_logging(args.verbose)

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
        _print_history(HistoryManager(max_size=config.history_size), args.history)
        return 0

    if args.print_config:
        print(config._render_config(), end="")
        return 0

    from sniptext.capture import ScreenCapture
    from sniptext.clipboard import ClipboardManager
    from sniptext.notify import send
    from sniptext.ocr import OCREngine

    # A capture bound to a key has no terminal: the notification is its only output.
    notify = bool(config.notification_enabled) and not args.file
    history_manager = (
        HistoryManager(max_size=config.history_size) if config.history_enabled else None
    )
    try:
        ocr_engine = OCREngine(config)
        clipboard_manager = ClipboardManager()

        if args.file:
            from PIL import Image, UnidentifiedImageError

            try:
                with Image.open(args.file) as opened:
                    image = opened.copy()
            except (OSError, UnidentifiedImageError) as e:
                logger.error(f"Failed to open image file '{args.file}': {e}")
                return 2
        else:
            logger.info("Capturing screen...")
            image = ScreenCapture(config).capture_region()
            if image is None:
                logger.error("Failed to capture screen")
                return 1

        text = ocr_engine.recognize(image)
        return _output_result(text, clipboard_manager, args.output, history_manager, notify)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception(e)
        if notify:
            send(f"✗ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
