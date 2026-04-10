"""
SnipText - Lightweight OCR Screen Capture Utility
Main entry point for the application.
"""

import argparse
import json
import sys
import urllib.request
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


def _capture_via_daemon(port: int = 9877) -> Optional[str]:
    """Connect to daemon and request capture. Returns text on success, None if error or cancelled."""
    url = f"http://localhost:{port}/capture"
    try:
        logger.info(f"Connecting to daemon at {url}...")
        request = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())
            if response.status == 200:
                text = data.get("text", "")
                logger.info(f"Captured {len(text)} chars via daemon")
                return text
            else:
                logger.error(f"Daemon error: {data.get('error', 'Unknown')}")
                return None
    except urllib.error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError):
            logger.error(f"Cannot connect to daemon at localhost:{port}")
            logger.error("Start daemon with: sniptext serve")
        else:
            logger.error(f"Failed to connect to daemon: {e.reason}")
        return None
    except urllib.error.HTTPError as e:
        try:
            error_data = json.loads(e.read().decode())
            if error_data.get("error") == "Capture cancelled or failed":
                logger.info("Capture cancelled")
                return None
            logger.error(f"Daemon HTTP error: {error_data.get('error')}")
        except (json.JSONDecodeError, OSError):
            logger.error(f"Daemon HTTP {e.code}: {e.reason}")
        return None
    except Exception as e:
        logger.error(f"Failed to communicate with daemon: {e}")
        return None


def _output_result(
    text: str,
    clipboard_manager,
    output_path: Optional[Path],
    history_manager=None,
    skip_clipboard: bool = False,
    skip_history: bool = False,
) -> int:
    """Print, copy, and optionally write OCR result. Returns exit code.

    Args:
        text: Recognized text
        clipboard_manager: Clipboard manager instance
        output_path: Optional file path to write text to
        history_manager: Optional history manager instance
        skip_clipboard: Skip clipboard copy (used when daemon already copied)
        skip_history: Skip history append (used when daemon already recorded)
    """
    if not text:
        print("✗ No text recognized")
        return 0

    print(text)

    if not skip_clipboard:
        copied = clipboard_manager.copy(text)
        if copied:
            print(f"\n✓ Copied {len(text)} characters to clipboard")
        else:
            logger.error("Failed to copy text to clipboard")
            return 1
    else:
        print(f"\n✓ {len(text)} characters (from daemon)")

    if output_path is not None:
        try:
            output_path.write_text(text, encoding="utf-8")
            print(f"✓ Saved to {output_path}")
        except OSError as e:
            logger.error(f"Failed to write output file: {e}")
            return 1

    if not skip_history and history_manager is not None:
        history_manager.append(text)

    return 0


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
        help=(
            "Apply a named config profile from PROFILES_DIR/NAME.yaml, where PROFILES_DIR is the "
            "'profiles' directory next to the config file (default: ~/.config/sniptext/profiles)"
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available config profiles and exit",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["serve"],
        help="Optional commands: 'serve' to start daemon mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9877,
        help="Port for daemon server (default: 9877)",
    )
    parser.add_argument(
        "--client",
        action="store_true",
        help="Connect to daemon for capture instead of running locally",
    )

    args = parser.parse_args()

    if args.output and not (args.file or args.capture_now):
        parser.error(
            "--output requires either --file or --capture-now; it is not supported in hotkey-only mode."
        )

    # Handle 'serve' command for daemon mode
    if args.command == "serve":
        from sniptext.config import Config
        from sniptext.daemon import SnipTextDaemon

        setup_logging(args.verbose)
        logger.info("Starting SnipText daemon...")

        if args.profile:
            try:
                config = Config.load_with_profile(args.config, args.profile)
            except FileNotFoundError as e:
                print(f"✗ {e}")
                return 1
            logger.info(f"Loaded config with profile {args.profile!r}")
        else:
            config = Config.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        daemon = SnipTextDaemon(config, port=args.port)
        try:
            daemon.start()
        except KeyboardInterrupt:
            logger.info("Daemon shutdown")
            return 0
        except Exception as e:
            logger.error(f"Daemon error: {e}")
            return 1
        return 0

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
            text = None
        elif args.capture_now and not args.client:
            screen_capture = ScreenCapture(config)
            logger.info("Capturing screen...")
            image = screen_capture.capture_region()
            if image is None:
                logger.error("Failed to capture screen")
                return 1
            text = None
        elif args.client and args.capture_now:
            # Use daemon client for capture
            text = _capture_via_daemon(args.port)
            if text is None:
                return 1
            image = None
        else:
            image = None
            text = None

        if text is not None:
            # Text from daemon client — already captured and copied
            rc = _output_result(
                text,
                clipboard_manager,
                args.output,
                history_manager,
                skip_clipboard=True,
                skip_history=True,
            )
            if rc != 0:
                return rc
        elif image is not None:
            # Local image to process
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
