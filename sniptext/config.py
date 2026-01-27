"""Configuration management for SnipText."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


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


    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not config_path.exists():
            # Create default config
            config = cls()
            config.save(config_path)
            return config

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        # Remove all old/unused parameters
        deprecated = [
            'preprocessing_enabled', 'preprocessing_mode', 'enable_text_correction',
            'save_history', 'history_db_path', 'max_history_items',
            'show_confidence_overlay', 'context_aware_detection', 'num_threads'
        ]
        for param in deprecated:
            data.pop(param, None)

        # Convert string paths to Path objects
        if 'ocr_model_path' in data and data['ocr_model_path']:
            data['ocr_model_path'] = Path(data['ocr_model_path']).expanduser()


        return cls(**data)

    def save(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert all data to YAML-safe types
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value

        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
