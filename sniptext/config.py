"""Configuration management for SnipText."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Application configuration."""

    # Hotkey configuration
    hotkey: str = "<ctrl>+<shift>+s"

    # Display server
    display_server: str = "auto"  # auto, wayland, x11

    # OCR configuration
    ocr_engine: str = "tesseract"  # tesseract (fast), easyocr (ML)
    ocr_model_path: Optional[Path] = None
    ocr_language: str = "en"  # en, ru, multi
    ocr_confidence_threshold: float = 0.6

    # Performance
    max_image_size: int = 4096  # max dimension
    use_gpu: bool = False
    num_threads: int = 4

    # Storage
    save_history: bool = True
    history_db_path: Optional[Path] = None
    max_history_items: int = 1000

    # UI
    show_confidence_overlay: bool = False
    notification_enabled: bool = True

    # Advanced
    preprocessing_enabled: bool = True
    context_aware_detection: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        if self.ocr_model_path is None:
            self.ocr_model_path = Path.home() / ".local" / "share" / "sniptext" / "models"

        if self.history_db_path is None:
            self.history_db_path = Path.home() / ".local" / "share" / "sniptext" / "history.db"

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

        # Convert string paths to Path objects
        if 'ocr_model_path' in data and data['ocr_model_path']:
            data['ocr_model_path'] = Path(data['ocr_model_path']).expanduser()

        if 'history_db_path' in data and data['history_db_path']:
            data['history_db_path'] = Path(data['history_db_path']).expanduser()

        return cls(**data)

    def save(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings for YAML
        data = self.__dict__.copy()
        if self.ocr_model_path:
            data['ocr_model_path'] = str(self.ocr_model_path)
        if self.history_db_path:
            data['history_db_path'] = str(self.history_db_path)

        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
