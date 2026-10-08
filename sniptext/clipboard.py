"""Clipboard management for SnipText."""

import shutil
import subprocess
from typing import Optional

from loguru import logger


class ClipboardManager:
    """Manages clipboard operations across different systems."""

    def __init__(self):
        """Initialize clipboard manager."""
        self._detect_clipboard_tool()

    def _detect_clipboard_tool(self) -> None:
        """Detect available clipboard tool."""
        # Try Wayland first
        if shutil.which("wl-copy"):
            self.tool = "wayland"
            self.copy_cmd = ["wl-copy"]
            logger.debug("Using wl-copy (Wayland)")
        # Fall back to X11
        elif shutil.which("xclip"):
            self.tool = "x11"
            self.copy_cmd = ["xclip", "-selection", "clipboard"]
            logger.debug("Using xclip (X11)")
        elif shutil.which("xsel"):
            self.tool = "x11"
            self.copy_cmd = ["xsel", "--clipboard", "--input"]
            logger.debug("Using xsel (X11)")
        else:
            raise RuntimeError(
                "No clipboard tool found. Please install wl-clipboard (Wayland) or xclip/xsel (X11)"
            )

    def copy(self, text: str) -> bool:
        """Copy text to the clipboard. True on success."""
        try:
            # The tool exits at once and leaves a child serving the selection. That child
            # inherits our pipes, so capturing output would block until the clipboard changes.
            result = subprocess.run(
                self.copy_cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error("Clipboard operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error copying to clipboard: {e}")
            return False
        if result.returncode != 0:
            logger.error(f"{self.copy_cmd[0]} failed with code {result.returncode}")
            return False
        logger.debug(f"Copied {len(text)} characters to clipboard")
        return True

    def paste(self) -> Optional[str]:
        """
        Get text from clipboard.

        Returns:
            Clipboard text or None if failed
        """
        try:
            if self.tool == "wayland":
                cmd = ["wl-paste"]
            elif self.tool == "x11":
                if "xclip" in self.copy_cmd[0]:
                    cmd = ["xclip", "-selection", "clipboard", "-o"]
                else:
                    cmd = ["xsel", "--clipboard", "--output"]
            else:
                return None

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                return result.stdout
            return None

        except Exception as e:
            logger.error(f"Error reading from clipboard: {e}")
            return None
