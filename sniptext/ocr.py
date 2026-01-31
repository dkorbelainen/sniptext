"""OCR engine for SnipText with multiple backends."""

import numpy as np
from PIL import Image
from loguru import logger
from abc import ABC, abstractmethod

from .config import Config
from .analyzer import ImageAnalyzer


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
        self._analyzer = ImageAnalyzer()

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
        """Recognize text using Tesseract with adaptive parameters."""
        if not self.is_available():
            raise RuntimeError("Tesseract not available")

        # Analyze image and optimize
        enhanced_image = self._analyzer.enhance_for_ocr(image)
        psm_mode = self._analyzer.suggest_psm_mode(image)

        lang_code = self._get_lang_code()
        custom_config = f'--oem 1 --psm {psm_mode}'

        logger.info(f"Tesseract: lang={lang_code}, PSM={psm_mode}")
        logger.debug(f"Using Tesseract with PSM {psm_mode}")

        text = self._tesseract.image_to_string(
            enhanced_image,
            lang=lang_code,
            config=custom_config
        )

        return text.strip()

    def _get_lang_code(self) -> str:
        """Get Tesseract language code."""
        return self.config.ocr_language


class EasyOCRBackend(OCRBackend):
    """EasyOCR backend (high accuracy)."""

    def __init__(self, config: Config):
        self.config = config
        self._reader = None
        self._available = None  # Lazy check
        self._initialized = False

    def _check_available(self) -> bool:
        """Check if EasyOCR is available (lazy check on first use)."""
        if self._available is not None:
            return self._available

        try:
            import easyocr
            self._available = True
            return True
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            self._available = False
            return False

    def is_available(self) -> bool:
        return self._check_available()

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

            # Lower threshold to catch more text (0.3 instead of 0.6)
            if confidence >= max(0.3, self.config.ocr_confidence_threshold * 0.5):
                lines.append(text)
                logger.debug(f"Text: '{text}' (confidence: {confidence:.2f})")
            else:
                logger.debug(f"Skipped low confidence: '{text}' ({confidence:.2f})")

        return "\n".join(lines)

    def _get_lang_codes(self) -> list[str]:
        """Get EasyOCR language codes."""
        # Convert Tesseract codes to EasyOCR codes
        lang_map = {
            'eng': 'en',
            'rus': 'ru',
            'jpn': 'ja',
            'kor': 'ko',
            'chi_sim': 'ch_sim',
            'chi_tra': 'ch_tra',
            'fra': 'fr',
            'deu': 'de',
            'spa': 'es',
        }

        # Split by + for multiple languages
        tess_langs = self.config.ocr_language.split('+')
        easy_langs = []

        for tl in tess_langs:
            tl = tl.strip()
            if tl in lang_map:
                easy_langs.append(lang_map[tl])
            else:
                # Try to use as-is (might work)
                easy_langs.append(tl)

        return easy_langs if easy_langs else ['en']


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

        # Lazy initialization of confidence model (only created when first used)
        self.confidence_model = None
        self._confidence_enabled = self.config.adaptive_ensemble and self.config.ocr_engine == "ensemble"

        backend_name = type(self.backend).__name__.replace("Backend", "").lower()
        logger.info(f"OCR Engine initialized with backend: {backend_name}")
        logger.info(f"OCR Language: {config.ocr_language}")
        logger.info(f"Text Correction: {config.enable_text_correction}")
        if self._confidence_enabled:
            logger.info(f"Adaptive ensemble: enabled")

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

    def _get_confidence_model(self):
        """Lazy initialization of confidence model (only when first needed)."""
        if self.confidence_model is None and self._confidence_enabled:
            from .confidence import ConfidenceModel
            self.confidence_model = ConfidenceModel()
        return self.confidence_model

    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize text from image using adaptive strategy.

        Args:
            image: Input image (numpy array)

        Returns:
            Recognized text
        """
        try:
            pil_image = self._prepare_image(image)

            # Check if we should use adaptive strategy
            confidence_model = self._get_confidence_model()
            if confidence_model and self.config.ocr_engine == "ensemble":
                # Predict optimal strategy
                strategy, confidence = confidence_model.predict_strategy(pil_image)

                if strategy == 'fast':
                    # Use single fast backend
                    logger.debug(f"Using fast mode (confidence: {confidence:.2f})")
                    text = self.backend.recognize(pil_image)
                else:
                    # Use ensemble for difficult images
                    logger.debug(f"Using ensemble mode (confidence: {confidence:.2f})")
                    text = self._recognize_ensemble(pil_image)

                if text:
                    logger.info(f"Recognized text: {len(text)} characters (strategy: {strategy})")
            elif self.config.ocr_engine == "ensemble":
                # Always use ensemble without adaptive selection
                text = self._recognize_ensemble(pil_image)
                if text:
                    logger.info(f"Recognized text: {len(text)} characters (mode: ensemble)")
            else:
                # Use single backend
                text = self.backend.recognize(pil_image)
                if text:
                    logger.info(f"Recognized text: {len(text)} characters")

            if text:
                logger.debug(f"Text preview: {text[:100]}...")

                if self.config.enable_text_correction:
                    from .ensemble import post_process_text
                    text = post_process_text(
                        text,
                        language=self.config.ocr_language,
                        enable_correction=True,
                        aggressive=self.config.aggressive_correction
                    )
            else:
                logger.debug("No text recognized")

            return text

        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return ""

    def _recognize_ensemble(self, image: Image.Image) -> str:
        """
        Recognize using ensemble of available backends.

        Args:
            image: PIL Image

        Returns:
            Combined text result
        """
        from .ensemble import EnsembleOCR

        results = []

        # Collect results from all available backends
        for name, backend in self.backends.items():
            if backend.is_available():
                try:
                    logger.debug(f"Running {name}...")
                    text = backend.recognize(image)
                    if text:
                        results.append(text)
                        logger.debug(f"{name}: {len(text)} chars")
                except Exception as e:
                    logger.warning(f"{name} failed: {e}")

        if not results:
            return ""

        if len(results) == 1:
            return results[0]

        # Combine results using ensemble
        ensemble = EnsembleOCR()
        combined = ensemble.combine_results(results)

        logger.info(f"Ensemble combined {len(results)} results")

        return combined


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
