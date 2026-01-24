"""Hotkey management for SnipText."""

import time
from typing import Callable
from loguru import logger
from pynput import keyboard

from .config import Config
from .capture import ScreenCapture
from .ocr import OCREngine
from .clipboard import ClipboardManager


class HotkeyManager:
    """Manages global hotkeys for screen capture."""

    def __init__(
        self,
        config: Config,
        screen_capture: ScreenCapture,
        ocr_engine: OCREngine,
        clipboard_manager: ClipboardManager,
    ):
        """
        Initialize hotkey manager.

        Args:
            config: Application configuration
            screen_capture: Screen capture instance
            ocr_engine: OCR engine instance
            clipboard_manager: Clipboard manager instance
        """
        self.config = config
        self.screen_capture = screen_capture
        self.ocr_engine = ocr_engine
        self.clipboard_manager = clipboard_manager

        self.listener = None
        self._parse_hotkey()

    def _parse_hotkey(self) -> None:
        """Parse hotkey string into modifier keys and key."""
        # Parse format like "<ctrl>+<shift>+s"
        parts = self.config.hotkey.lower().replace(" ", "").split("+")

        self.modifiers = set()
        self.key = None

        for part in parts:
            part = part.strip("<>")
            if part in ("ctrl", "control"):
                self.modifiers.add(keyboard.Key.ctrl_l)
                self.modifiers.add(keyboard.Key.ctrl_r)
            elif part in ("shift",):
                self.modifiers.add(keyboard.Key.shift_l)
                self.modifiers.add(keyboard.Key.shift_r)
            elif part in ("alt",):
                self.modifiers.add(keyboard.Key.alt_l)
                self.modifiers.add(keyboard.Key.alt_r)
            elif part in ("super", "win", "meta"):
                self.modifiers.add(keyboard.Key.cmd)
            else:
                # Regular key
                self.key = part

        logger.debug(f"Parsed hotkey - modifiers: {self.modifiers}, key: {self.key}")

    def start(self) -> None:
        """Start listening for hotkeys."""
        current_keys = set()

        def on_press(key):
            """Handle key press."""
            try:
                # Add to current keys
                if hasattr(key, 'char') and key.char:
                    current_keys.add(key.char.lower())
                else:
                    current_keys.add(key)

                # Check if hotkey is pressed
                if self._is_hotkey_pressed(current_keys):
                    logger.info("Hotkey pressed!")
                    self._on_hotkey_triggered()

            except Exception as e:
                logger.error(f"Error in key press handler: {e}")

        def on_release(key):
            """Handle key release."""
            try:
                # Remove from current keys
                if hasattr(key, 'char') and key.char:
                    current_keys.discard(key.char.lower())
                else:
                    current_keys.discard(key)

            except Exception as e:
                logger.error(f"Error in key release handler: {e}")

        # Start keyboard listener
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            self.listener = listener
            listener.join()

    def _is_hotkey_pressed(self, current_keys: set) -> bool:
        """
        Check if the hotkey combination is pressed.

        Args:
            current_keys: Set of currently pressed keys

        Returns:
            True if hotkey is pressed
        """
        # Check if all modifiers are pressed
        modifiers_pressed = any(mod in current_keys for mod in self.modifiers) if self.modifiers else True

        # For modifiers, we need at least one from each group
        if self.modifiers:
            ctrl_mods = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
            shift_mods = {keyboard.Key.shift_l, keyboard.Key.shift_r}
            alt_mods = {keyboard.Key.alt_l, keyboard.Key.alt_r}

            needs_ctrl = bool(self.modifiers & ctrl_mods)
            needs_shift = bool(self.modifiers & shift_mods)
            needs_alt = bool(self.modifiers & alt_mods)

            has_ctrl = bool(current_keys & ctrl_mods)
            has_shift = bool(current_keys & shift_mods)
            has_alt = bool(current_keys & alt_mods)

            modifiers_ok = (
                (not needs_ctrl or has_ctrl) and
                (not needs_shift or has_shift) and
                (not needs_alt or has_alt)
            )
        else:
            modifiers_ok = True

        # Check if main key is pressed
        key_pressed = self.key in current_keys if self.key else False

        return modifiers_ok and key_pressed

    def _on_hotkey_triggered(self) -> None:
        """Handle hotkey trigger - capture and OCR."""
        start_time = time.time()

        try:
            # Capture screen
            logger.info("Capturing screen region...")
            image = self.screen_capture.capture_region()

            if image is None:
                logger.warning("Screen capture cancelled or failed")
                return

            capture_time = time.time() - start_time
            logger.debug(f"Capture took {capture_time:.3f}s")

            # Run OCR
            logger.info("Running OCR...")
            ocr_start = time.time()
            text = self.ocr_engine.recognize(image)
            ocr_time = time.time() - ocr_start
            logger.debug(f"OCR took {ocr_time:.3f}s")

            if not text:
                logger.warning("No text recognized")
                return

            # Copy to clipboard
            logger.info("Copying to clipboard...")
            success = self.clipboard_manager.copy(text)

            if success:
                total_time = time.time() - start_time
                logger.info(
                    f"Recognized {len(text)} characters in {total_time:.3f}s "
                    f"(capture: {capture_time:.3f}s, OCR: {ocr_time:.3f}s)"
                )

                # Show notification if enabled
                if self.config.notification_enabled:
                    self._show_notification(f"Copied {len(text)} characters")
            else:
                logger.error("Failed to copy to clipboard")

        except Exception as e:
            logger.error(f"Error processing capture: {e}")
            logger.exception(e)

    def _show_notification(self, message: str) -> None:
        """
        Show desktop notification.

        Args:
            message: Notification message
        """
        try:
            import subprocess
            subprocess.run(
                ["notify-send", "SnipText", message],
                timeout=2,
                capture_output=True,
            )
        except Exception as e:
            logger.debug(f"Could not show notification: {e}")

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self.listener:
            self.listener.stop()
