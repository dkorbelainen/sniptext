# SnipText

Screen capture OCR tool for Linux. Select screen area, get text in clipboard.

Useful for extracting text from images, PDFs, videos, or any non-selectable content without manual retyping.

## Features

- **Adaptive Ensemble OCR**: Automatic selection between fast and accurate modes
- **Intelligent merging**: Combines Tesseract and EasyOCR results using smart heuristics
- **Simple workflow**: Hotkey → Select area → Text copied
- **Multi-language**: Supports any Tesseract/EasyOCR language
- **Works on Wayland and X11**
- **GPU acceleration**: Uses CUDA when available

## Installation

### Arch Linux (AUR)

```bash
yay -S sniptext
# or: paru -S sniptext
```

After installation, you'll see instructions for setting up a keyboard shortcut.

### Other Linux Distributions

#### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng python3-pip
# Add languages: sudo apt install tesseract-ocr-rus tesseract-ocr-fra etc.

# Wayland:
sudo apt install slurp grim wl-clipboard
# X11:
sudo apt install maim xclip
```

**Fedora:**
```bash
sudo dnf install tesseract tesseract-langpack-eng python3-pip
# Wayland:
sudo dnf install slurp grim wl-clipboard
# X11:
sudo dnf install maim xclip
```

#### 2. Install SnipText

```bash
pip install --user sniptext
# or system-wide:
sudo pip install sniptext
```

#### 3. Setup Keyboard Shortcut

⚠️ **REQUIRED: Manual keybind setup**

SnipText requires you to configure a keyboard shortcut in your window manager or desktop environment.

**Recommended keybind: `Ctrl+Alt+T`**

Configure your system to run this command when the keybind is pressed:
```bash
sniptext --capture-now
```

#### Configuration Examples:

**Hyprland** (`~/.config/hypr/hyprland.conf`):
```
bind = CTRL ALT, T, exec, sniptext --capture-now
```

**Sway** (`~/.config/sway/config`):
```
bindsym Ctrl+Alt+t exec sniptext --capture-now
```

**i3** (`~/.config/i3/config`):
```
bindsym Ctrl+Alt+t exec sniptext --capture-now
```

**GNOME**: Settings → Keyboard → Keyboard Shortcuts → Custom Shortcuts
- Click "+" to add new shortcut
- Name: `SnipText OCR`
- Command: `sniptext --capture-now`
- Set shortcut: Press `Ctrl+Alt+T`

**KDE Plasma**: System Settings → Shortcuts → Custom Shortcuts
- Right-click → New → Global Shortcut → Command/URL
- Name: `SnipText OCR`
- Trigger: Set to `Ctrl+Alt+T`
- Action: `sniptext --capture-now`

Feel free to use any keybind you prefer (e.g., `Super+Shift+S`).

**Why manual setup?** On Wayland, applications cannot register global hotkeys due to security restrictions. Even on X11, WM-level shortcuts are more reliable than application-level hotkeys.

## Usage

1. Press your configured keybind (e.g., `Ctrl+Alt+T`)
2. Select screen area with mouse
3. Text is automatically copied to clipboard

**Test from terminal:**
```bash
sniptext --capture-now
```

## Language Support

SnipText uses Tesseract OCR, which supports 100+ languages.

**Arch Linux:**
```bash
# Install additional language packs
sudo pacman -S tesseract-data-rus    # Russian
sudo pacman -S tesseract-data-jpn    # Japanese
sudo pacman -S tesseract-data-chi-sim # Chinese Simplified
# See all: pacman -Ss tesseract-data
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr-rus tesseract-ocr-jpn
```

Then update your config file to use the language(s):
```yaml
ocr_language: eng+rus  # Multiple languages
```

## Configuration

Config file: `~/.config/sniptext/config.yaml` (created on first run)

```yaml
# OCR engine mode
ocr_engine: ensemble        # ensemble (best), tesseract, easyocr
adaptive_ensemble: true     # auto fast/accurate mode selection

# Language - use Tesseract language codes
ocr_language: eng           # English only
# ocr_language: rus         # Russian only
# ocr_language: eng+rus     # English + Russian
# ocr_language: jpn         # Japanese
# See all codes: tesseract --list-langs

# Other settings
notification_enabled: true
use_gpu: true               # GPU acceleration (auto-detects CUDA)
```

**Adaptive Ensemble**: When enabled with `ensemble` mode, a machine learning model analyzes each image and automatically chooses:
- **Fast mode** (Tesseract only) for clear, high-contrast screenshots
- **Ensemble mode** (both engines) for difficult, low-contrast images

This gives you both speed and accuracy without manual switching.

GPU is enabled by default - automatically uses CUDA if available, falls back to CPU if not.
