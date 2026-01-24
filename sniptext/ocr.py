"""OCR engine for SnipText with multiple backends."""

import numpy as np
from PIL import Image
from loguru import logger
from abc import ABC, abstractmethod

from .config import Config
from .postprocess import TextPostprocessor


class OCRBackend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def recognize(self, image: Image.Image) -> str:
        """Recognize text from image."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class TesseractBackend(OCRBackend):
    """Tesseract OCR backend (fast, lightweight)."""

    def __init__(self, config: Config):
        self.config = config
        self._tesseract = None
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            self._tesseract = pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    def recognize(self, image: Image.Image) -> str:
        """Recognize text using Tesseract."""
        if not self.is_available():
            raise RuntimeError("Tesseract not available")

        lang_code = self._get_lang_code()
        # OEM 3: LSTM only, PSM 6: Assume uniform block of text
        custom_config = r'--oem 1 --psm 6'

        text = self._tesseract.image_to_string(
            image,
            lang=lang_code,
            config=custom_config
        )

        return text.strip()

    def _get_lang_code(self) -> str:
        """Get Tesseract language code."""
        lang_map = {
            "en": "eng",
            "ru": "rus",
            "multi": "eng+rus",
        }
        return lang_map.get(self.config.ocr_language, "eng")


class EasyOCRBackend(OCRBackend):
    """EasyOCR backend (high accuracy)."""

    def __init__(self, config: Config):
        self.config = config
        self._reader = None
        self._available = self._check_available()
        self._initialized = False

    def _check_available(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            return True
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            return False

    def is_available(self) -> bool:
        return self._available

    def _lazy_init(self):
        """Lazy initialization of EasyOCR reader."""
        if self._initialized:
            return

        try:
            import easyocr

            langs = self._get_lang_codes()

            logger.info(f"Initializing EasyOCR with languages: {langs}")
            logger.info("First run will download models (~100-500MB)...")

            self._reader = easyocr.Reader(
                langs,
                gpu=self.config.use_gpu,
                verbose=False
            )

            self._initialized = True
            logger.info("EasyOCR initialized")

        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def recognize(self, image: Image.Image) -> str:
        """Recognize text using EasyOCR."""
        if not self.is_available():
            raise RuntimeError("EasyOCR not available")

        self._lazy_init()

        img_array = np.array(image)

        results = self._reader.readtext(
            img_array,
            detail=1,
            paragraph=False
        )

        lines = []
        for detection in results:
            bbox, text, confidence = detection

            if confidence >= self.config.ocr_confidence_threshold:
                lines.append(text)
                logger.debug(f"Text: '{text}' (confidence: {confidence:.2f})")

        return "\n".join(lines)

    def _get_lang_codes(self) -> list:
        """Get EasyOCR language codes."""
        lang_map = {
            "en": ["en"],
            "ru": ["ru"],
            "multi": ["en", "ru"],
        }
        return lang_map.get(self.config.ocr_language, ["en"])


class OCREngine:
    """Main OCR engine with multiple backend support."""

    def __init__(self, config: Config):
        """
        Initialize OCR engine.

        Args:
            config: Application configuration
        """
        self.config = config
        self.backends = {
            "tesseract": TesseractBackend(config),
            "easyocr": EasyOCRBackend(config),
        }
        self.backend = self._initialize_backend()
        self.postprocessor = TextPostprocessor()

        backend_name = type(self.backend).__name__.replace("Backend", "").lower()
        logger.info(f"OCR Engine initialized with backend: {backend_name}")

    def _initialize_backend(self) -> OCRBackend:
        """Initialize the appropriate OCR backend."""
        engine_name = self.config.ocr_engine.lower()

        if engine_name == "easyocr":
            backend = self.backends["easyocr"]
            if not backend.is_available():
                logger.warning("EasyOCR not available, falling back to Tesseract")
                backend = self.backends["tesseract"]
        else:
            backend = self.backends["tesseract"]
            if not backend.is_available():
                logger.warning("Tesseract not available, trying EasyOCR")
                backend = self.backends["easyocr"]

        if not backend.is_available():
            raise RuntimeError(
                "No OCR backend available. Install either:\n"
                "  - Tesseract: sudo pacman -S tesseract\n"
                "  - EasyOCR: pip install easyocr"
            )

        return backend

    def get_available_backends(self) -> list[str]:
        """Get list of available backend names."""
        return [name for name, backend in self.backends.items() if backend.is_available()]

    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize text from image.

        Args:
            image: Input image (numpy array)

        Returns:
            Recognized text
        """
        try:
            pil_image = self._prepare_image(image)

            # Apply preprocessing if enabled
            if self.config.preprocessing_enabled:
                pil_image = self._preprocess_image(pil_image)

            text = self.backend.recognize(pil_image)

            # Apply postprocessing
            if text and self.config.preprocessing_enabled:
                text = self.postprocessor.process(text, self.config.ocr_language)

            if text:
                logger.info(f"Recognized text: {len(text)} characters")
                logger.debug(f"Text preview: {text[:100]}...")
            else:
                logger.debug("No text recognized")

            return text

        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return ""

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input PIL Image

        Returns:
            Preprocessed PIL Image
        """
        from PIL import ImageEnhance, ImageOps, ImageFilter
        import cv2

        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Convert to numpy for opencv processing
        img_array = np.array(image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

        # Convert back to PIL
        image = Image.fromarray(denoised)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)

        logger.debug("Applied adaptive preprocessing")
        return image

    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.

        Args:
            image: Numpy array image

        Returns:
            PIL Image
        """
        if isinstance(image, Image.Image):
            return image

        if len(image.shape) == 2:
            return Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                return Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:
                return Image.fromarray(image, mode='RGBA')

        logger.warning(f"Unexpected image shape: {image.shape}")
        return Image.fromarray(image)
