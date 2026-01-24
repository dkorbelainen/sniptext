"""Screen capture functionality for SnipText."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
import tempfile
import numpy as np
from PIL import Image
from loguru import logger

from .config import Config


class ScreenCapture:
    """Handles screen capture operations."""

    def __init__(self, config: Config):
        """
        Initialize screen capture.

        Args:
            config: Application configuration
        """
        self.config = config
        self._detect_display_server()
        self._detect_capture_tools()

    def _detect_display_server(self) -> None:
        """Detect the display server (Wayland or X11)."""
        if self.config.display_server != "auto":
            self.display_server = self.config.display_server
            logger.info(f"Using configured display server: {self.display_server}")
            return

        # Auto-detect
        wayland_display = subprocess.run(
            ["printenv", "WAYLAND_DISPLAY"],
            capture_output=True,
            text=True,
        ).stdout.strip()

        if wayland_display:
            self.display_server = "wayland"
            logger.info("Detected Wayland display server")
        else:
            self.display_server = "x11"
            logger.info("Detected X11 display server")

    def _detect_capture_tools(self) -> None:
        """Detect available screenshot tools."""
        if self.display_server == "wayland":
            # Check for Wayland tools
            if shutil.which("slurp") and shutil.which("grim"):
                self.capture_method = "slurp_grim"
                logger.debug("Using slurp + grim for Wayland")
            elif shutil.which("grimshot"):
                self.capture_method = "grimshot"
                logger.debug("Using grimshot for Wayland")
            else:
                raise RuntimeError(
                    "No Wayland screenshot tool found. Please install slurp + grim or grimshot"
                )
        else:
            # Check for X11 tools
            if shutil.which("maim"):
                self.capture_method = "maim"
                logger.debug("Using maim for X11")
            elif shutil.which("scrot"):
                self.capture_method = "scrot"
                logger.debug("Using scrot for X11")
            elif shutil.which("import"):  # ImageMagick
                self.capture_method = "import"
                logger.debug("Using ImageMagick import for X11")
            else:
                raise RuntimeError(
                    "No X11 screenshot tool found. Please install maim, scrot, or imagemagick"
                )

    def capture_region(self, output_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        Capture a screen region selected by the user.

        Args:
            output_path: Optional path to save the screenshot

        Returns:
            Captured image as numpy array or None if failed
        """
        # Use temp file if no output path specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
            cleanup_temp = True
        else:
            cleanup_temp = False

        try:
            # Capture screenshot
            success = self._capture_to_file(output_path)

            if not success or not output_path.exists():
                logger.error("Screenshot capture failed")
                return None

            # Load image
            image = Image.open(output_path)
            image_array = np.array(image)

            logger.info(f"Captured image: {image_array.shape}")

            # Cleanup temp file
            if cleanup_temp:
                output_path.unlink()

            return image_array

        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            if cleanup_temp and output_path.exists():
                output_path.unlink()
            return None

    def _capture_to_file(self, output_path: Path) -> bool:
        """
        Capture screenshot to file using detected method.

        Args:
            output_path: Path to save the screenshot

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.capture_method == "slurp_grim":
                # Get selection geometry with slurp
                slurp_result = subprocess.run(
                    ["slurp"],
                    capture_output=True,
                    text=True,
                    timeout=60,  # Wait for user selection
                )

                if slurp_result.returncode != 0:
                    logger.warning("User cancelled selection")
                    return False

                geometry = slurp_result.stdout.strip()

                # Capture with grim
                grim_result = subprocess.run(
                    ["grim", "-g", geometry, str(output_path)],
                    capture_output=True,
                    timeout=10,
                )

                return grim_result.returncode == 0

            elif self.capture_method == "grimshot":
                result = subprocess.run(
                    ["grimshot", "save", "area", str(output_path)],
                    capture_output=True,
                    timeout=60,
                )
                return result.returncode == 0

            elif self.capture_method == "maim":
                result = subprocess.run(
                    ["maim", "-s", str(output_path)],
                    capture_output=True,
                    timeout=60,
                )
                return result.returncode == 0

            elif self.capture_method == "scrot":
                result = subprocess.run(
                    ["scrot", "-s", str(output_path)],
                    capture_output=True,
                    timeout=60,
                )
                return result.returncode == 0

            elif self.capture_method == "import":
                result = subprocess.run(
                    ["import", str(output_path)],
                    capture_output=True,
                    timeout=60,
                )
                return result.returncode == 0

            else:
                logger.error(f"Unknown capture method: {self.capture_method}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Screenshot capture timed out")
            return False
        except Exception as e:
            logger.error(f"Error in capture: {e}")
            return False
