"""
SnipText - Lightweight OCR Screen Capture Utility
Main entry point for the application.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

from sniptext.capture import ScreenCapture
from sniptext.ocr import OCREngine
from sniptext.clipboard import ClipboardManager
from sniptext.config import Config
from sniptext.hotkey import HotkeyManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
    )


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="SnipText - OCR Screen Capture Utility"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path.home() / ".config" / "sniptext" / "config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
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
        help="OCR engine: tesseract (fast), easyocr (accurate), ensemble (best quality)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger.info("Starting SnipText...")

    # Load configuration
    config = Config.load(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Override OCR engine if specified
    if args.ocr_engine:
        config.ocr_engine = args.ocr_engine
        logger.info(f"OCR engine overridden to: {args.ocr_engine}")


    # Initialize components
    try:
        screen_capture = ScreenCapture(config)
        ocr_engine = OCREngine(config)
        clipboard_manager = ClipboardManager()


        logger.info("Components initialized successfully")

        if args.capture_now:
            # Single capture mode
            logger.info("Capturing screen...")
            image = screen_capture.capture_region()

            if image is not None:
                logger.info("Running OCR...")

                # Ensemble mode: run both engines and combine
                if args.ocr_engine == "ensemble":
                    from sniptext.ensemble import EnsembleOCR, post_process_text
                    from sniptext.ocr import TesseractBackend, EasyOCRBackend

                    logger.info("Running ensemble mode (Tesseract + EasyOCR)...")

                    results = []

                    # Run Tesseract
                    tesseract = TesseractBackend(config)
                    if tesseract.is_available():
                        logger.info("Running Tesseract...")
                        from PIL import Image as PILImage
                        pil_img = PILImage.fromarray(image)
                        text_tess = tesseract.recognize(pil_img)
                        results.append(text_tess)
                        logger.info(f"Tesseract: {len(text_tess)} chars")

                    # Run EasyOCR
                    easyocr = EasyOCRBackend(config)
                    if easyocr.is_available():
                        logger.info("Running EasyOCR...")
                        from PIL import Image as PILImage
                        pil_img = PILImage.fromarray(image)
                        text_easy = easyocr.recognize(pil_img)
                        results.append(text_easy)
                        logger.info(f"EasyOCR: {len(text_easy)} chars")

                    # Combine results
                    ensemble = EnsembleOCR()
                    text = ensemble.combine_results(results)
                    confidence = ensemble.calculate_confidence(results)

                    # Post-process
                    text = post_process_text(text)

                    logger.info(f"Ensemble result: {len(text)} chars, confidence: {confidence:.2%}")
                else:
                    text = ocr_engine.recognize(image)

                if text:
                    logger.info(f"Recognized text ({len(text)} chars)")
                    clipboard_manager.copy(text)
                    logger.info("Text copied to clipboard")

                    print(text)
                else:
                    logger.warning("No text recognized")
            else:
                logger.error("Failed to capture screen")
                return 1
        else:
            # Hotkey daemon mode
            logger.info("Starting hotkey daemon...")
            hotkey_manager = HotkeyManager(
                config=config,
                screen_capture=screen_capture,
                ocr_engine=ocr_engine,
                clipboard_manager=clipboard_manager,
            )

            logger.info(f"Press {config.hotkey} to capture and OCR")
            logger.info("Press Ctrl+C to exit")

            hotkey_manager.start()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.exception(e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
