"""Tests for ImageAnalyzer."""

import numpy as np
import pytest
from PIL import Image

from sniptext.analyzer import ImageAnalyzer


@pytest.fixture
def analyzer():
    return ImageAnalyzer()


def make_image(width, height, value=150, mode="RGB"):
    """Create a solid-color test image."""
    if mode == "L":
        return Image.fromarray(np.full((height, width), value, dtype=np.uint8), mode="L")
    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8))


class TestExtractFeatures:
    def test_returns_seven_features(self, analyzer):
        img = make_image(300, 100)
        features = analyzer.extract_features(img)
        assert len(features) == 7

    def test_all_features_in_range(self, analyzer):
        img = make_image(300, 100)
        features = analyzer.extract_features(img)
        for f in features:
            assert 0.0 <= f <= 1.0

    def test_bright_image_high_brightness(self, analyzer):
        img = make_image(300, 100, value=240)
        features = analyzer.extract_features(img)
        assert features[0] > 0.8  # brightness

    def test_dark_image_low_brightness(self, analyzer):
        img = make_image(300, 100, value=20)
        features = analyzer.extract_features(img)
        assert features[0] < 0.2

    def test_grayscale_image(self, analyzer):
        img = make_image(300, 100, mode="L")
        features = analyzer.extract_features(img)
        assert len(features) == 7

    def test_rgba_image(self, analyzer):
        arr = np.full((100, 300, 4), 150, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        features = analyzer.extract_features(img)
        assert len(features) == 7

    def test_text_density_high_for_dark_image(self, analyzer):
        """Mostly-dark image should have high text_density (many dark pixels)."""
        img = make_image(300, 100, value=20)
        features = analyzer.extract_features(img)
        assert features[5] > 0.8  # text_density

    def test_text_density_low_for_white_image(self, analyzer):
        """Mostly-white image should have low text_density (few dark pixels)."""
        img = make_image(300, 100, value=240)
        features = analyzer.extract_features(img)
        assert features[5] < 0.1  # text_density

    def test_noise_level_low_for_uniform_image(self, analyzer):
        """Solid-color image has no high-freq variation — noise should be near 0."""
        img = make_image(300, 100, value=150)
        features = analyzer.extract_features(img)
        assert features[6] < 0.1  # noise_level

    def test_noise_level_high_for_noisy_image(self, analyzer):
        """Synthetic noisy image should have significantly higher noise_level than a uniform one."""
        # Baseline: solid-color image
        uniform_img = make_image(300, 100, value=150)
        uniform_features = analyzer.extract_features(uniform_img)

        # Synthetic noise: random RGB pixels
        rng = np.random.RandomState(0)
        noisy_array = rng.randint(0, 256, size=(100, 300, 3), dtype=np.uint8)
        noisy_img = Image.fromarray(noisy_array, mode="RGB")
        noisy_features = analyzer.extract_features(noisy_img)

        # Noise level should be noticeably higher for the noisy image
        assert noisy_features[6] > uniform_features[6] + 0.2
        assert noisy_features[6] > 0.3


class TestSuggestPsmMode:
    def test_wide_short_image_returns_psm7(self, analyzer):
        # aspect ratio > 4 and height < 100
        img = make_image(500, 50)
        assert analyzer.suggest_psm_mode(img) == 7

    def test_small_image_returns_psm11(self, analyzer):
        img = make_image(200, 80)
        assert analyzer.suggest_psm_mode(img) == 11

    def test_tall_narrow_returns_psm6(self, analyzer):
        # aspect ratio < 0.5, but width >= 300 to skip the small-image branch
        img = make_image(300, 800)
        assert analyzer.suggest_psm_mode(img) == 6

    def test_normal_block_returns_psm6(self, analyzer):
        img = make_image(600, 400)
        assert analyzer.suggest_psm_mode(img) == 6


class TestShouldInvert:
    def test_dark_image_should_invert(self, analyzer):
        img = make_image(300, 100, value=30)
        assert analyzer.should_invert(img) is True

    def test_bright_image_no_invert(self, analyzer):
        img = make_image(300, 100, value=200)
        assert analyzer.should_invert(img) is False


class TestEnhanceForOcr:
    def test_returns_pil_image(self, analyzer):
        img = make_image(400, 200)
        result = analyzer.enhance_for_ocr(img)
        assert isinstance(result, Image.Image)

    def test_small_image_gets_upscaled(self, analyzer):
        img = make_image(100, 50)
        result = analyzer.enhance_for_ocr(img)
        assert result.width > img.width
        assert result.height > img.height

    def test_degenerate_image_does_not_crash(self, analyzer):
        """A 0-size image must not raise ZeroDivisionError."""
        img = Image.new("RGB", (0, 0))
        result = analyzer.enhance_for_ocr(img)
        assert isinstance(result, Image.Image)

    def test_dark_image_not_over_brightened(self, analyzer):
        """After inverting a dark image brightness must stay reasonable (not > 240)."""
        img = make_image(400, 200, value=40)
        result = analyzer.enhance_for_ocr(img)
        import numpy as np

        brightness = np.array(result).mean()
        assert brightness < 240, f"Over-brightened: {brightness:.1f}"

    def test_low_contrast_image_gets_enhanced(self, analyzer):
        """A low-contrast image (stddev ≈ 20) must trigger contrast enhancement."""
        # Build a grayscale image with small intensity variations so that the
        # contrast is low but non-zero, and verify that enhancement increases it.
        width, height = 400, 200
        base = 150
        data = np.full((height, width), base, dtype=np.uint8)
        # Introduce a subtle checkerboard-like pattern around the base value.
        data[:, ::2] = base - 10  # 140
        data[:, 1::2] = base + 10  # 160
        img = Image.fromarray(data, mode="L")

        input_std = np.array(img).std()
        assert input_std > 0

        result = analyzer.enhance_for_ocr(img)
        # Post-enhancement the result is still a valid PIL image.
        assert isinstance(result, Image.Image)

        output_std = np.array(result.convert("L")).std()
        assert output_std > input_std
