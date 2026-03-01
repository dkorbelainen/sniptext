"""Clipboard management for SnipText."""

import shutil
import subprocess
import time
from typing import Optional

from loguru import logger


class ClipboardManager:
    """Manages clipboard operations across different systems."""

    def __init__(self):
        """Initialize clipboard manager."""
        self._wl_process: Optional[subprocess.Popen] = None
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
        """
        Copy text to clipboard.

        Args:
            text: Text to copy

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.tool == "wayland":
                # Kill the previous wl-copy process before spawning a new one;
                # wl-copy stays alive to serve the clipboard selection and
                # repeated copies would otherwise accumulate orphaned processes.
                if self._wl_process is not None and self._wl_process.poll() is None:
                    self._wl_process.terminate()
                    self._wl_process = None

                # wl-copy stays running (serves clipboard) until the content
                # is replaced; we must NOT use communicate() here or we'd
                # block until the user pastes.
                process = subprocess.Popen(
                    ["wl-copy"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                try:
                    process.stdin.write(text.encode("utf-8"))
                    process.stdin.close()
                except BrokenPipeError:
                    # wl-copy exited before we finished writing
                    stderr = process.stderr.read().decode(errors="replace")
                    logger.error(f"wl-copy closed unexpectedly: {stderr}")
                    return False

                # Give compositor a moment to register the new selection
                time.sleep(0.05)

                # Check if it started successfully — wl-copy must still be
                # running to keep the Wayland clipboard selection alive.
                # Any early exit (even with returncode 0) means the selection
                # won't be served.
                if process.poll() is not None:
                    stderr = process.stderr.read().decode(errors="replace")
                    logger.error(f"wl-copy exited unexpectedly (rc={process.returncode}): {stderr}")
                    return False

                self._wl_process = process
                logger.debug(
                    f"Copied {len(text)} characters to clipboard (wl-copy pid: {process.pid})"
                )
                return True
            else:
                # X11 tools work synchronously
                process = subprocess.Popen(
                    self.copy_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = process.communicate(input=text.encode("utf-8"), timeout=2)

                if process.returncode != 0:
                    logger.error(f"Failed to copy to clipboard: {stderr.decode()}")
                    return False

                logger.debug(f"Copied {len(text)} characters to clipboard")
                return True

        except subprocess.TimeoutExpired:
            logger.error("Clipboard operation timed out")
            return False
        except Exception as e:
            logger.error(f"Error copying to clipboard: {e}")
            return False

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
