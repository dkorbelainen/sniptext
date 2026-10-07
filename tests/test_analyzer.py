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


def text_image(theme="light", font_size=20, width=640, height=200):
    """Render a few lines of text with Pillow's built-in scalable font."""
    from PIL import ImageDraw, ImageFont

    bg, fg = (
        ((250, 250, 250), (20, 20, 20)) if theme == "light" else ((30, 30, 30), (212, 212, 212))
    )
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(font_size)
    y = 12
    while y < height - font_size - 8:
        draw.text((12, y), "The quick brown fox jumps over 0123456789", font=font, fill=fg)
        y += font_size + 10
    return img


class TestExtractFeatures:
    def test_length_matches_feature_names(self, analyzer):
        from sniptext.analyzer import FEATURE_NAMES

        assert len(FEATURE_NAMES) == 12
        assert len(set(FEATURE_NAMES)) == 12
        assert analyzer.extract_features(make_image(300, 100)).shape == (12,)

    def test_all_features_in_range(self, analyzer):
        for img in (make_image(300, 100), text_image("light"), text_image("dark")):
            features = analyzer.extract_features(img)
            assert np.all(np.isfinite(features))
            assert np.all((features >= 0.0) & (features <= 1.0))

    def test_degenerate_images_give_finite_features(self, analyzer):
        for size in ((1, 1), (15, 3), (3, 40)):
            features = analyzer.extract_features(Image.new("RGB", size, (128, 128, 128)))
            assert features.shape == (12,)
            assert np.all(np.isfinite(features))
            assert np.all((features >= 0.0) & (features <= 1.0))

    def test_bright_image_high_brightness(self, analyzer):
        assert analyzer.extract_features(make_image(300, 100, value=240))[0] > 0.8

    def test_dark_image_low_brightness(self, analyzer):
        assert analyzer.extract_features(make_image(300, 100, value=20))[0] < 0.2

    def test_grayscale_and_rgba_inputs(self, analyzer):
        assert analyzer.extract_features(make_image(300, 100, mode="L")).shape == (12,)
        rgba = Image.fromarray(np.full((100, 300, 4), 150, dtype=np.uint8), mode="RGBA")
        assert analyzer.extract_features(rgba).shape == (12,)

    def test_text_density_is_theme_invariant(self, analyzer):
        """The old feature counted dark pixels, so a dark theme measured background."""
        light = analyzer.extract_features(text_image("light"))[5]
        dark = analyzer.extract_features(text_image("dark"))[5]
        assert 0.0 < light < 0.5
        assert abs(light - dark) < 0.03

    def test_text_height_grows_with_font_size(self, analyzer):
        small = analyzer.extract_features(text_image(font_size=14))[10]
        large = analyzer.extract_features(text_image(font_size=30))[10]
        assert large > small > 0.0

    def test_blur_lowers_sharpness(self, analyzer):
        from PIL import ImageFilter

        img = text_image()
        sharp = analyzer.extract_features(img)[2]
        blurred = analyzer.extract_features(img.filter(ImageFilter.GaussianBlur(1.5)))[2]
        assert blurred < sharp < 1.0

    def test_noise_level_low_for_uniform_image(self, analyzer):
        assert analyzer.extract_features(make_image(300, 100, value=150))[6] < 0.05

    def test_noise_raises_noise_level(self, analyzer):
        img = text_image()
        arr = np.asarray(img).astype(np.float32)
        noisy = np.clip(arr + np.random.default_rng(0).normal(0, 30, arr.shape), 0, 255)
        clean_level = analyzer.extract_features(img)[6]
        noisy_level = analyzer.extract_features(Image.fromarray(noisy.astype(np.uint8)))[6]
        assert noisy_level > clean_level + 0.1

    def test_jpeg_raises_blockiness(self, analyzer):
        import io

        img = text_image()
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=5)
        buf.seek(0)
        clean = analyzer.extract_features(img)[11]
        compressed = analyzer.extract_features(Image.open(buf).convert("RGB"))[11]
        assert compressed > clean

    def test_extraction_time_bound(self, analyzer):
        import time

        img = text_image(width=1920, height=1080)
        analyzer.extract_features(img)
        # Best of five: a busy machine stalls single runs, a slow path stalls all of them.
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            analyzer.extract_features(img)
            best = min(best, time.perf_counter() - start)
        assert best < 0.5


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

    def test_bright_image_gets_dimmed(self, analyzer):
        """A very bright image (brightness > 200) must trigger brightness reduction."""
        img = make_image(400, 200, value=240)
        result = analyzer.enhance_for_ocr(img)
        brightness_out = np.array(result.convert("L")).mean()
        # Sharpening can slightly change brightness, but the result must be
        # meaningfully darker than the raw 240 input.
        assert brightness_out < 240
