"""OCR engine benchmarking and comparison utility."""

import time
from pathlib import Path
from typing import Dict, Optional, TypedDict

from loguru import logger
from PIL import Image

from .config import Config
from .ocr import OCREngine


class BenchmarkResult(TypedDict, total=False):
    """Result structure for a single engine benchmark."""

    time_seconds: float
    chars_recognized: int
    status: str


class OCRBenchmark:
    """Benchmark OCR engines for speed and accuracy."""

    def __init__(self, config: Config):
        """Initialize benchmark.

        Args:
            config: Application configuration
        """
        self.config = config
        self.engine = OCREngine(config)
        self.results: Dict[str, Dict[str, BenchmarkResult]] = {}

    def benchmark_file(self, image_path: Path) -> Optional[Dict[str, BenchmarkResult]]:
        """Benchmark all available OCR engines on a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary of results: {engine_name: {time_seconds: float, chars_recognized: int, status: str}}
        """
        try:
            pil_image = Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

        results: Dict[str, BenchmarkResult] = {}

        # Test each backend
        for backend_name, backend in self.engine.backends.items():
            if not backend.is_available():
                results[backend_name] = {"status": "unavailable"}  # type: ignore
                continue

            try:
                logger.info(f"Benchmarking {backend_name}...")

                start_time = time.perf_counter()
                text = backend.recognize(pil_image)
                elapsed = time.perf_counter() - start_time

                results[backend_name] = {
                    "time_seconds": round(elapsed, 3),
                    "chars_recognized": len(text),
                    "status": "ok",
                }
                logger.info(f"  {backend_name}: {elapsed:.3f}s, {len(text)} chars")

            except Exception as e:
                logger.error(f"  {backend_name} failed: {e}")
                results[backend_name] = {"status": f"error: {e}"}

        self.results[str(image_path)] = results
        return results

    def benchmark_directory(
        self, dir_path: Path, pattern: str = "*.png"
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Benchmark engines on multiple images in a directory.

        Args:
            dir_path: Directory containing images
            pattern: File pattern to match (default: "*.png")

        Returns:
            Dictionary of results for each image
        """
        if not dir_path.is_dir():
            logger.error(f"Directory not found: {dir_path}")
            return {}

        images = list(dir_path.glob(pattern))
        if not images:
            logger.warning(f"No images matching {pattern} in {dir_path}")
            return {}

        logger.info(f"Found {len(images)} images to benchmark")

        all_results = {}
        for image_path in sorted(images):
            logger.info(f"Processing {image_path.name}...")
            result = self.benchmark_file(image_path)
            if result:
                all_results[image_path.name] = result

        return all_results

    def print_summary(self) -> None:
        """Print benchmark results summary."""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "=" * 80)
        print("OCR ENGINE BENCHMARK RESULTS")
        print("=" * 80 + "\n")

        for image_name, engines_results in self.results.items():
            print(f"Image: {image_name}")
            print("-" * 80)

            # Find fastest and slowest
            valid_results = {
                name: data for name, data in engines_results.items() if data.get("status") == "ok"
            }

            if valid_results:
                fastest = min(valid_results.items(), key=lambda x: x[1]["time_seconds"])
                slowest = max(valid_results.items(), key=lambda x: x[1]["time_seconds"])

                for engine_name, result in sorted(engines_results.items()):
                    status = result.get("status", "unknown")

                    if status == "ok":
                        time_s = result["time_seconds"]
                        chars = result["chars_recognized"]

                        # Mark fastest/slowest
                        marker = ""
                        if engine_name == fastest[0]:
                            marker = " ⚡ FASTEST"
                        elif engine_name == slowest[0]:
                            marker = " 🐌 SLOWEST"

                        print(f"  {engine_name:12s}: {time_s:7.3f}s  {chars:4d} chars{marker}")
                    elif status == "unavailable":
                        print(f"  {engine_name:12s}: [not installed]")
                    else:
                        print(f"  {engine_name:12s}: [error] {status}")

            print()

        print("=" * 80)
        print("Note: Times shown are for engine processing only.")
        print("First run may be slower due to model initialization.\n")
