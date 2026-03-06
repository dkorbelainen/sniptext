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
        if not isinstance(self.ocr_engine, str) or self.ocr_engine not in valid_engines:
            logger.warning(
                f"Invalid ocr_engine={self.ocr_engine!r}; must be one of {sorted(valid_engines)}. "
                "Resetting to 'ensemble'."
            )
            self.ocr_engine = "ensemble"

        valid_display = {"auto", "wayland", "x11"}
        if not isinstance(self.display_server, str) or self.display_server not in valid_display:
            logger.warning(
                f"Invalid display_server={self.display_server!r}; must be one of {sorted(valid_display)}. "
                "Resetting to 'auto'."
            )
            self.display_server = "auto"

        try:
            threshold_ok = 0.0 < float(self.ocr_confidence_threshold) <= 1.0
        except (TypeError, ValueError):
            threshold_ok = False
        if not threshold_ok:
            logger.warning(
                f"Invalid ocr_confidence_threshold={self.ocr_confidence_threshold!r}; "
                "must be a number in (0, 1]. Resetting to 0.6."
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

    def _render_config(self) -> str:
        """Render configuration as a YAML string with inline comments."""
        _COMMENTS: dict[str, str] = {
            "hotkey": "Global hotkey to trigger capture (e.g. <ctrl>+<alt>+t)",
            "display_server": "Display server: auto, wayland, or x11",
            "ocr_engine": "OCR engine: ensemble (recommended), tesseract, or easyocr",
            "ocr_model_path": "Directory for EasyOCR model files (leave blank for default)",
            "ocr_language": "Tesseract language code(s), e.g. eng, rus, eng+rus",
            "ocr_confidence_threshold": "Minimum OCR confidence to accept a result (0.0–1.0)",
            "adaptive_ensemble": "Auto-select fast/ensemble mode based on image quality",
            "max_image_size": "Resize images larger than this (pixels) before OCR",
            "use_gpu": "Use GPU acceleration for EasyOCR when available (requires CUDA)",
            "notification_enabled": "Show desktop notification after each capture",
            "enable_text_correction": "Apply automatic spell/OCR error correction",
            "aggressive_correction": "More aggressive correction (may introduce errors)",
        }

        lines: list[str] = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            serialized = yaml.dump({f.name: value}, default_flow_style=False).rstrip()
            comment = _COMMENTS.get(f.name)
            if comment:
                lines.append(f"# {comment}")
            lines.append(serialized)

        return "\n".join(lines) + "\n"

    def save(self, config_path: Path) -> None:
        """Save configuration to YAML file with inline comments."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self._render_config())
