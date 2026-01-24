"""Image analyzer for OCR optimization."""

import numpy as np
from PIL import Image, ImageStat
from loguru import logger


class ImageAnalyzer:
    """Analyze image and suggest optimal OCR parameters."""

    def __init__(self):
        """Initialize analyzer."""
        self.features = []


    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from image for analysis.

        Returns:
            Feature vector: [brightness, contrast, sharpness, has_color, size_ratio]
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Calculate statistics
        stat = ImageStat.Stat(image)

        # Brightness (0-255)
        brightness = np.mean(stat.mean)

        # Contrast (std deviation)
        contrast = np.mean(stat.stddev)

        # Estimate sharpness using edge detection
        gray = image.convert('L')
        img_array = np.array(gray)
        laplacian = np.abs(np.diff(img_array, axis=0)).mean() + np.abs(np.diff(img_array, axis=1)).mean()
        sharpness = laplacian

        # Color presence
        r, g, b = stat.mean
        color_variance = np.std([r, g, b])
        has_color = 1 if color_variance > 10 else 0

        # Size (normalized)
        width, height = image.size
        size_ratio = width / height if height > 0 else 1.0

        # Better normalization
        # Contrast: typical range 10-80 for text images
        normalized_contrast = min(contrast / 60.0, 1.0)

        # Sharpness: typical range 5-50 for text images
        normalized_sharpness = min(sharpness / 30.0, 1.0)

        features = np.array([
            brightness / 255.0,      # normalize to 0-1
            normalized_contrast,     # normalize to 0-1
            normalized_sharpness,    # normalize to 0-1
            has_color,               # binary
            min(size_ratio, 5.0) / 5.0  # cap at 5:1 ratio
        ])

        logger.debug(f"Image features: brightness={brightness:.1f}, contrast={contrast:.1f}, "
                    f"sharpness={sharpness:.1f}, color={has_color}, ratio={size_ratio:.2f}")

        return features

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
        features = self.extract_features(image)
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
        features = self.extract_features(image)
        brightness = features[0] * 255

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
        from PIL import ImageEnhance

        # Upscale small images - critical for OCR quality
        width, height = image.size
        if height < 100 or width < 300:
            scale_factor = max(2.0, 100 / height, 300 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.debug(f"Upscaled image from {width}x{height} to {new_width}x{new_height}")

        features = self.extract_features(image)
        brightness = features[0] * 255
        contrast = features[1] * 128

        # Invert if dark background
        if self.should_invert(image):
            from PIL import ImageOps
            image = ImageOps.invert(image.convert('RGB'))
            logger.debug("Applied color inversion")

        # Enhance contrast if low
        if contrast < 40:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.8)
            logger.debug("Applied contrast enhancement")

        # Sharpen for better edge detection
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        logger.debug("Applied sharpening")

        # Adjust brightness if needed
        if brightness < 80:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.4)
            logger.debug("Applied brightness increase")
        elif brightness > 200:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)
            logger.debug("Applied brightness decrease")

        return image
