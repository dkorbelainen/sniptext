# SnipText

Screen capture OCR tool for Linux. Select screen area, get text in clipboard.

Useful for extracting text from images, PDFs, videos, or any non-selectable content without manual retyping.

## Features

- **Ensemble OCR**: Combines Tesseract and EasyOCR for better accuracy
- **Simple workflow**: Hotkey → Select area → Text copied
- **Multi-language**: Supports any Tesseract/EasyOCR language
- **Works on Wayland and X11**

## Installation

### 1. System Dependencies

**Arch Linux:**
```bash
# Base OCR engine + your language data
sudo pacman -S tesseract tesseract-data-eng

# Add other languages as needed:
# sudo pacman -S tesseract-data-rus  # Russian
# sudo pacman -S tesseract-data-chi-sim  # Chinese Simplified
# sudo pacman -S tesseract-data-jpn  # Japanese
# See: pacman -Ss tesseract-data

# Screen capture tools
sudo pacman -S slurp grim wl-clipboard  # Wayland
# OR for X11:
# sudo pacman -S maim xclip
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
# Add languages: sudo apt install tesseract-ocr-rus tesseract-ocr-fra etc.

# Wayland:
sudo apt install slurp grim wl-clipboard
# X11:
sudo apt install maim xclip
```

### 2. Python Setup

```bash
git clone https://github.com/dkorbelainen/sniptext
cd sniptext

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For ensemble mode:
pip install easyocr torch torchvision
```

### 3. Global Hotkey

Add to your window manager config (replace `/path/to` with actual path):

**Hyprland** (`~/.config/hypr/hyprland.conf`):
```
bind = CTRL ALT, T, exec, /path/to/sniptext/quick-ocr.sh
```

**i3/Sway** (`~/.config/i3/config` or `~/.config/sway/config`):
```
bindsym Ctrl+Alt+t exec /path/to/sniptext/quick-ocr.sh
```

**KDE/GNOME**: Use System Settings → Keyboard Shortcuts

Feel free to change the hotkey if you feel like doing so.
Reload config after adding hotkey.

## Usage

**With hotkey:** Press `Ctrl+Alt+T`, select area with mouse

**Manual:** `cd sniptext && ./quick-ocr.sh`

## Configuration

Config file: `~/.config/sniptext/config.yaml` (created on first run)

```yaml
# OCR engine mode
ocr_engine: ensemble   # ensemble (best), tesseract, easyocr

# Language - use Tesseract language codes
ocr_language: eng      # English only
# ocr_language: rus      # Russian only
# ocr_language: eng+rus  # English + Russian
# ocr_language: jpn      # Japanese
# See all codes: tesseract --list-langs

# Other settings
notification_enabled: true
use_gpu: true          # GPU acceleration (auto-detects CUDA)
```

GPU is enabled by default - automatically uses CUDA if available, falls back to CPU if not.

**Common language codes:**
- `eng` - English
- `rus` - Russian
- `fra` - French
- `deu` - German
- `spa` - Spanish
- `chi_sim` - Chinese Simplified
- `jpn` - Japanese
- `kor` - Korean

Combine with `+`: `eng+fra+deu`

## How Ensemble Works

The ensemble mode combines two OCR engines for better accuracy:

1. **Capture** screen region
2. **Process** image (auto brightness/contrast/upscaling)
3. **Run both** Tesseract and EasyOCR
4. **Compare** results line-by-line
5. **Score each variant:**
   - Line completeness (length)
   - Proper punctuation
   - Character quality
   - Language-specific features
6. **Select** best variant per line
7. **Post-process** (fix spacing, punctuation)

Ensemble typically produces better results by choosing the best output from each engine.

