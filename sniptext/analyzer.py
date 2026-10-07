"""Image analyzer for OCR optimization."""

import numpy as np
from loguru import logger
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

FEATURE_NAMES = (
    "brightness",
    "contrast",
    "sharpness",
    "has_color",
    "size_ratio",
    "text_density",
    "noise_level",
    "width",
    "height",
    "edge_density",
    "text_height",
    "jpeg_blockiness",
)


def _otsu_threshold(gray: np.ndarray) -> int:
    """Threshold maximising between-class variance of an 8-bit image."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    below = np.cumsum(hist)
    above = hist.sum() - below
    mass = np.cumsum(hist * np.arange(256))
    mean_below = np.divide(mass, below, out=np.zeros(256), where=below > 0)
    mean_above = np.divide(mass[-1] - mass, above, out=np.zeros(256), where=above > 0)
    return int(np.argmax(below * above * (mean_below - mean_above) ** 2))


def _median_text_row_run(foreground: np.ndarray) -> float:
    """Median height of the runs of rows that contain foreground pixels."""
    rows = foreground.mean(axis=1) > 0.005
    if not rows.any():
        return 0.0
    edges = np.diff(np.concatenate(([0], rows.astype(np.int8), [0])))
    return float(np.median(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)))


class ImageAnalyzer:
    """Analyze image and suggest optimal OCR parameters."""

    def __init__(self):
        """Initialize analyzer."""
        self.features = []

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract the routing features, in FEATURE_NAMES order, each in [0, 1].
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        if width == 0 or height == 0:
            return np.zeros(len(FEATURE_NAMES))

        stat = ImageStat.Stat(image)
        brightness = float(np.mean(stat.mean))
        contrast = float(np.mean(stat.stddev))
        red, green, blue = stat.mean
        has_color = 1.0 if np.std([red, green, blue]) > 10 else 0.0

        gray_image = image.convert("L")
        gray = np.asarray(gray_image)
        # int16: differences of uint8 wrap around.
        levels = gray.astype(np.int16)
        dy = np.abs(np.diff(levels, axis=0))
        dx = np.abs(np.diff(levels, axis=1))
        sharpness = (float(dy.mean()) if dy.size else 0.0) + (float(dx.mean()) if dx.size else 0.0)

        dark = gray <= _otsu_threshold(gray)
        dark_share = float(dark.mean())
        # Text is the minority class on either theme.
        foreground = dark if dark_share <= 0.5 else ~dark
        text_density = min(dark_share, 1.0 - dark_share)

        smoothed = np.asarray(gray_image.filter(ImageFilter.SMOOTH), dtype=np.int16)
        residual = levels - smoothed
        # Median absolute deviation: glyph edges are sparse outliers, noise is everywhere.
        noise_level = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))

        if dx.size and dy.size:
            edge_density = float(((dx[:-1, :] > 32) | (dy[:, :-1] > 32)).mean())
        else:
            edge_density = 0.0

        if width >= 16:
            on_boundary = (np.arange(dx.shape[1]) % 8) == 7
            ratio = float(dx[:, on_boundary].mean()) / (float(dx[:, ~on_boundary].mean()) + 1e-6)
            jpeg_blockiness = min(ratio, 4.0) / 4.0
        else:
            jpeg_blockiness = 0.25

        features = np.array(
            [
                brightness / 255.0,
                min(contrast / 60.0, 1.0),
                min(sharpness / 128.0, 1.0),
                has_color,
                min(width / height, 5.0) / 5.0,
                text_density,
                min(noise_level / 64.0, 1.0),
                min(width / 4096.0, 1.0),
                min(height / 4096.0, 1.0),
                edge_density,
                min(_median_text_row_run(foreground) / 64.0, 1.0),
                jpeg_blockiness,
            ]
        )
        logger.debug(
            "Image features: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(FEATURE_NAMES, features))
        )
        return features

    def _brightness_contrast(self, image: Image.Image) -> tuple:
        """Mean brightness (0-255) and mean channel stddev, without the full feature pass."""
        stat = ImageStat.Stat(image.convert("RGB"))
        return float(np.mean(stat.mean)), float(np.mean(stat.stddev))

    def suggest_psm_mode(self, image: Image.Image) -> int:
        """
        Suggest Tesseract PSM mode based on image analysis.

        PSM modes:
        3 = Fully automatic page segmentation (default)
        6 = Uniform block of text
        7 = Single text line
        11 = Sparse text
        13 = Raw line (for single line)

        Returns:
            PSM mode number
        """
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0

        # Decision tree based on image characteristics
        if aspect_ratio > 4.0 and height < 100:
            # Very wide, short image - likely single line
            logger.debug("Detected single line of text (PSM 7)")
            return 7
        elif width < 300 or height < 100:
            # Small image - sparse text
            logger.debug("Detected sparse text (PSM 11)")
            return 11
        elif aspect_ratio < 0.5:
            # Tall narrow image - might be vertical text or column
            logger.debug("Detected narrow column (PSM 6)")
            return 6
        else:
            # Normal text block
            logger.debug("Detected text block (PSM 6)")
            return 6

    def should_invert(self, image: Image.Image) -> bool:
        """
        Check if image should be inverted (dark background).

        Returns:
            True if should invert colors
        """
        brightness, _ = self._brightness_contrast(image)

        # If image is dark (< 100 brightness), likely dark background
        if brightness < 100:
            logger.debug(f"Dark image detected (brightness={brightness:.1f}), suggesting inversion")
            return True

        return False

    def enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Apply optimal enhancements based on image analysis.

        Args:
            image: Input image

        Returns:
            Enhanced image
        """
        # Upscale small images - critical for OCR quality
        width, height = image.size
        if width == 0 or height == 0:
            logger.warning(f"Received degenerate image ({width}x{height}), skipping enhancement")
            return image
        if height < 100 or width < 300:
            scale_factor = max(2.0, 100 / height, 300 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.debug(f"Upscaled image from {width}x{height} to {new_width}x{new_height}")

        brightness, contrast = self._brightness_contrast(image)

        # Invert if dark background (reuse already-computed brightness)
        inverted = False
        if brightness < 100:
            image = ImageOps.invert(image.convert("RGB"))
            inverted = True
            logger.debug(f"Dark image detected (brightness={brightness:.1f}), applied inversion")

        # Enhance contrast if low
        if contrast < 40:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.8)
            logger.debug("Applied contrast enhancement")

        # Sharpen for better edge detection
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        logger.debug("Applied sharpening")

        # Reduce brightness for over-exposed images.
        # Skip this step for inverted images: the brightness metric and threshold
        # are based on the original (pre-inversion) image, and we don't want to
        # further darken content that started out on a dark background.
        if not inverted and brightness > 200:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)
            logger.debug("Applied brightness decrease")

        return image
