"""Hotkey management for SnipText."""

import os
import subprocess
import threading
import time

from loguru import logger

from .capture import ScreenCapture
from .clipboard import ClipboardManager
from .config import Config
from .ocr import OCREngine


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
        # threading.Event provides atomic is_set/set/clear across threads,
        # avoiding the read-check-then-spawn race that a plain bool would have.
        self._processing = threading.Event()
        self._parse_hotkey()

    def _parse_hotkey(self) -> None:
        """Parse hotkey string into modifier keys and key."""
        from pynput import keyboard

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
                self.key = part

        logger.debug(f"Parsed hotkey - modifiers: {self.modifiers}, key: {self.key}")

    def start(self) -> None:
        """Start listening for hotkeys."""
        # WAYLAND_DISPLAY is more reliable than XDG_SESSION_TYPE — the latter
        # is not always set (e.g. some Sway/Hyprland setups), while the former
        # is always present when a Wayland compositor is running.
        is_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        if is_wayland:
            logger.warning("=" * 60)
            logger.warning("WAYLAND DETECTED - Hotkeys may not work!")
            logger.warning("=" * 60)
            logger.warning("pynput cannot capture global hotkeys on Wayland.")
            logger.warning("Workaround: Use a keyboard shortcut in your DE/WM to run:")
            logger.warning("  sniptext --capture-now")
            logger.warning("")
            logger.warning("Example for Hyprland (~/.config/hypr/hyprland.conf):")
            logger.warning("  bind = SUPER SHIFT, S, exec, sniptext --capture-now")
            logger.warning("")
            logger.warning("Example for Sway (~/.config/sway/config):")
            logger.warning("  bindsym $mod+Shift+s exec sniptext --capture-now")
            logger.warning("")
            logger.warning("Example for GNOME:")
            logger.warning("  Settings → Keyboard → Custom Shortcuts")
            logger.warning("=" * 60)

        current_keys = set()

        def on_press(key):
            """Handle key press."""
            try:
                if hasattr(key, "char") and key.char:
                    current_keys.add(key.char.lower())
                else:
                    current_keys.add(key)

                if self._is_hotkey_pressed(current_keys):
                    logger.info("Hotkey pressed!")
                    if self._processing.is_set():
                        logger.debug("Already processing a capture, ignoring hotkey")
                        return
                    # Run OCR in a background thread so the listener thread
                    # is not blocked and remains responsive.
                    thread = threading.Thread(target=self._on_hotkey_triggered, daemon=True)
                    thread.start()

            except Exception as e:
                logger.error(f"Error in key press handler: {e}")

        def on_release(key):
            """Handle key release."""
            try:
                if hasattr(key, "char") and key.char:
                    current_keys.discard(key.char.lower())
                else:
                    current_keys.discard(key)

            except Exception as e:
                logger.error(f"Error in key release handler: {e}")

        try:
            from pynput import keyboard

            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                self.listener = listener
                listener.join()
        except Exception as e:
            logger.error("=" * 60)
            logger.error("FAILED TO START KEYBOARD LISTENER")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            logger.error("")
            logger.error("This usually happens on Wayland or without proper permissions.")
            logger.error("")
            logger.error("Solutions:")
            logger.error("1. Use --capture-now flag instead of hotkey mode")
            logger.error("2. Bind a keyboard shortcut in your DE/WM to run:")
            logger.error("     sniptext --capture-now")
            logger.error("3. If on X11, ensure your user has input permissions")
            logger.error("=" * 60)
            raise

    def _is_hotkey_pressed(self, current_keys: set) -> bool:
        """
        Check if the hotkey combination is pressed.

        Args:
            current_keys: Set of currently pressed keys

        Returns:
            True if hotkey is pressed
        """
        from pynput import keyboard

        if self.modifiers:
            ctrl_mods = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
            shift_mods = {keyboard.Key.shift_l, keyboard.Key.shift_r}
            alt_mods = {keyboard.Key.alt_l, keyboard.Key.alt_r}
            super_mods = {keyboard.Key.cmd}

            needs_ctrl = bool(self.modifiers & ctrl_mods)
            needs_shift = bool(self.modifiers & shift_mods)
            needs_alt = bool(self.modifiers & alt_mods)
            needs_super = bool(self.modifiers & super_mods)

            has_ctrl = bool(current_keys & ctrl_mods)
            has_shift = bool(current_keys & shift_mods)
            has_alt = bool(current_keys & alt_mods)
            has_super = bool(current_keys & super_mods)

            modifiers_ok = (
                (not needs_ctrl or has_ctrl)
                and (not needs_shift or has_shift)
                and (not needs_alt or has_alt)
                and (not needs_super or has_super)
            )
        else:
            modifiers_ok = True

        key_pressed = self.key in current_keys if self.key else False

        return modifiers_ok and key_pressed

    def _on_hotkey_triggered(self) -> None:
        """Handle hotkey trigger - capture and OCR (runs in a background thread)."""
        self._processing.set()
        start_time = time.time()

        try:
            logger.info("Capturing screen region...")
            image = self.screen_capture.capture_region()

            if image is None:
                logger.warning("Screen capture cancelled or failed")
                return

            capture_time = time.time() - start_time
            logger.debug(f"Capture took {capture_time:.3f}s")

            logger.info("Running OCR...")
            ocr_start = time.time()
            text = self.ocr_engine.recognize(image)
            ocr_time = time.time() - ocr_start
            logger.debug(f"OCR took {ocr_time:.3f}s")

            if not text:
                logger.warning("No text recognized")
                return

            logger.info("Copying to clipboard...")
            success = self.clipboard_manager.copy(text)

            if success:
                total_time = time.time() - start_time
                logger.info(
                    f"Recognized {len(text)} characters in {total_time:.3f}s "
                    f"(capture: {capture_time:.3f}s, OCR: {ocr_time:.3f}s)"
                )

                if self.config.notification_enabled:
                    self._show_notification(f"Copied {len(text)} characters")
            else:
                logger.error("Failed to copy to clipboard")

        except Exception as e:
            logger.error(f"Error processing capture: {e}")
            logger.exception(e)
        finally:
            self._processing.clear()

    def _show_notification(self, message: str) -> None:
        """
        Show desktop notification.

        Args:
            message: Notification message
        """
        try:
            result = subprocess.run(
                ["notify-send", "SnipText", message],
                timeout=2,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                logger.debug(f"notify-send failed with code {result.returncode}")
        except FileNotFoundError:
            logger.debug("notify-send not found - install libnotify package")
        except Exception as e:
            logger.debug(f"Could not show notification: {e}")

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self.listener:
            self.listener.stop()
