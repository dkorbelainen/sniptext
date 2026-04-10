"""Tests for sniptext.benchmark OCR engine benchmarking."""

from unittest.mock import patch

from PIL import Image

from sniptext.benchmark import OCRBenchmark
from sniptext.config import Config


class TestOCRBenchmark:
    """Tests for OCRBenchmark class."""

    def test_benchmark_init(self, tmp_path):
        """Test benchmark initialization."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        assert benchmark.config == config
        assert benchmark.engine is not None
        assert benchmark.results == {}

    def test_benchmark_file_invalid_path(self, tmp_path):
        """Test benchmark with invalid image path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        invalid_path = tmp_path / "nonexistent.png"
        result = benchmark.benchmark_file(invalid_path)

        assert result is None

    @patch("sniptext.benchmark.OCRBenchmark.print_summary")
    def test_benchmark_file_success(self, mock_print, tmp_path):
        """Test successful benchmark on valid image."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        # Create a simple test image
        image_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(image_path)

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        result = benchmark.benchmark_file(image_path)

        # Result should be a dict with backend names as keys
        if result:
            for backend_name, data in result.items():
                assert isinstance(backend_name, str)
                assert isinstance(data, dict)

    def test_print_summary_empty(self, tmp_path, capsys):
        """Test print_summary with no results."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        benchmark.print_summary()
        captured = capsys.readouterr()

        assert "No results" in captured.out

    def test_print_summary_with_results(self, tmp_path, capsys):
        """Test print_summary with mock results."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        # Add mock results
        benchmark.results = {
            "test.png": {
                "tesseract": {"status": "ok", "time_seconds": 1.234, "chars_recognized": 100},
                "easyocr": {"status": "unavailable"},
            }
        }

        benchmark.print_summary()
        captured = capsys.readouterr()

        assert "OCR ENGINE BENCHMARK RESULTS" in captured.out
        assert "test.png" in captured.out
        assert "tesseract" in captured.out

    def test_benchmark_directory_not_found(self, tmp_path):
        """Test benchmark with nonexistent directory."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        invalid_dir = tmp_path / "nonexistent"
        result = benchmark.benchmark_directory(invalid_dir)

        assert result == {}

    def test_benchmark_directory_no_images(self, tmp_path):
        """Test benchmark with directory containing no images."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        image_dir = tmp_path / "images"
        image_dir.mkdir()

        # Create non-image files
        (image_dir / "readme.txt").write_text("test")

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        result = benchmark.benchmark_directory(image_dir, pattern="*.png")

        assert result == {}

    @patch.object(OCRBenchmark, "benchmark_file")
    def test_benchmark_directory_with_images(self, mock_bench_file, tmp_path):
        """Test benchmark with multiple images."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("ocr_engine: tesseract\ndisplay_server: x11\nhotkey: ctrl+alt+s\n")

        image_dir = tmp_path / "images"
        image_dir.mkdir()

        # Create test images
        for i in range(2):
            img = Image.new("RGB", (50, 50), color="white")
            img.save(image_dir / f"test{i}.png")

        config = Config.load(config_file)
        benchmark = OCRBenchmark(config)

        # Mock benchmark_file to return results
        mock_bench_file.return_value = {
            "tesseract": {"status": "ok", "time_seconds": 0.1, "chars_recognized": 50}
        }

        result = benchmark.benchmark_directory(image_dir, pattern="*.png")

        # Should have called benchmark_file for each image
        assert mock_bench_file.call_count == 2
        assert len(result) == 2
