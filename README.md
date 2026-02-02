# SnipText

Screen text extractor with OCR and spell correction for Arch Linux.

**Simple workflow:** Hotkey → Select area → Text in clipboard

## Features

- **Adaptive OCR**: Automatic mode selection for speed and accuracy
- **Multi-language**: 100+ languages via Tesseract (English, Russian, Greek, Math symbols, etc.)
- **Spell correction**: Automatic text correction for better results
- **Works everywhere**: Wayland and X11 support
- **GPU acceleration**: CUDA support when available

## Installation

```bash
yay -S sniptext
```

**Important:** Set up a keyboard shortcut after installation. See [KEYBINDINGS.md](KEYBINDINGS.md)

## Usage

1. Press your keybind (default: `Ctrl+Alt+T`)
2. Select screen area
3. Text copied to clipboard

**Test:** `sniptext --capture-now`

## Language Support

Install additional language packs:

```bash
sudo pacman -S tesseract-data-rus     # Russian
sudo pacman -S tesseract-data-ell     # Greek
sudo pacman -S tesseract-data-equ     # Math symbols
# See all: pacman -Ss tesseract-data
```

Update config: `~/.config/sniptext/config.yaml`
```yaml
ocr_language: eng+rus+equ  # English + Russian + Math
```

**Full guide:** [LANGUAGES.md](LANGUAGES.md)

## Configuration

Config: `~/.config/sniptext/config.yaml` (auto-created on first run)

```yaml
ocr_engine: ensemble        # ensemble, tesseract, or easyocr
ocr_language: eng           # See LANGUAGES.md for codes
adaptive_ensemble: true     # Auto quality-based mode selection
enable_text_correction: true
notification_enabled: true
use_gpu: true               # CUDA acceleration
```

**Adaptive Ensemble:** Automatically uses fast mode for clear images, accurate mode for difficult ones.

**Note:** Optional features (spell correction, EasyOCR, ML analysis) are auto-detected when installed.

## Optional Dependencies

All optional dependencies are auto-detected. Install to enable features:

```bash
yay -S python-symspellpy      # Spell correction (English)
yay -S python-scikit-learn    # Quality analysis
yay -S python-easyocr         # High-accuracy OCR (slower, GPU recommended)
```