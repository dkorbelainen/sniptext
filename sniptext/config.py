"""Configuration management for SnipText."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


@dataclass
class Config:
    """Application configuration."""

    # Hotkey configuration
    hotkey: str = "<ctrl>+<alt>+t"

    # Display server
    display_server: str = "auto"  # auto, wayland, x11

    # OCR configuration
    ocr_engine: str = "ensemble"  # ensemble, tesseract, easyocr
    ocr_model_path: Optional[Path] = None
    ocr_language: str = "eng"  # Language code (eng, rus, eng+rus, etc.)
    ocr_confidence_threshold: float = 0.6
    adaptive_ensemble: bool = True  # Automatically choose fast/ensemble mode based on image quality

    # Performance
    max_image_size: int = 4096
    use_gpu: bool = True  # Use GPU if available (CUDA for EasyOCR)

    # UI
    notification_enabled: bool = True

    # Text correction
    enable_text_correction: bool = True  # Apply OCR error corrections
    aggressive_correction: bool = False  # Apply more aggressive corrections (may introduce errors)

    def __post_init__(self):
        """Post-initialization setup."""
        if self.ocr_model_path is None:
            self.ocr_model_path = Path.home() / ".local" / "share" / "sniptext" / "models"

        self._validate()

    def _validate(self) -> None:
        """Validate config values, resetting invalid ones to defaults with a warning."""
        valid_engines = {"ensemble", "tesseract", "easyocr"}
        if self.ocr_engine not in valid_engines:
            logger.warning(
                f"Invalid ocr_engine={self.ocr_engine!r}; must be one of {sorted(valid_engines)}. "
                "Resetting to 'ensemble'."
            )
            self.ocr_engine = "ensemble"

        valid_display = {"auto", "wayland", "x11"}
        if self.display_server not in valid_display:
            logger.warning(
                f"Invalid display_server={self.display_server!r}; must be one of {sorted(valid_display)}. "
                "Resetting to 'auto'."
            )
            self.display_server = "auto"

        if not (0.0 < self.ocr_confidence_threshold <= 1.0):
            logger.warning(
                f"Invalid ocr_confidence_threshold={self.ocr_confidence_threshold!r}; "
                "must be in (0, 1]. Resetting to 0.6."
            )
            self.ocr_confidence_threshold = 0.6

        if not isinstance(self.max_image_size, int) or self.max_image_size < 64:
            logger.warning(
                f"Invalid max_image_size={self.max_image_size!r}; "
                "must be an integer >= 64. Resetting to 4096."
            )
            self.max_image_size = 4096

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not config_path.exists():
            config = cls()
            config.save(config_path)
            return config

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Remove all old/unused parameters
        deprecated = [
            "preprocessing_enabled",
            "preprocessing_mode",
            "save_history",
            "history_db_path",
            "max_history_items",
            "show_confidence_overlay",
            "context_aware_detection",
            "num_threads",
        ]
        for param in deprecated:
            data.pop(param, None)

        # Convert string paths to Path objects
        if "ocr_model_path" in data and data["ocr_model_path"]:
            data["ocr_model_path"] = Path(data["ocr_model_path"]).expanduser()

        # Drop unknown keys to avoid TypeError instead of crashing
        known_keys = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known_keys
        if unknown:
            logger.warning(f"Ignoring unknown config keys: {sorted(unknown)}")
            data = {k: v for k, v in data.items() if k in known_keys}

        return cls(**data)

    def save(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            f.name: str(v) if isinstance(v := getattr(self, f.name), Path) else v
            for f in dataclasses.fields(self)
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
