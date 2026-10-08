# SnipText

[![CI](https://github.com/dkorbelainen/sniptext/actions/workflows/ci.yml/badge.svg)](https://github.com/dkorbelainen/sniptext/actions/workflows/ci.yml)

Select a region of the screen; Tesseract reads it and the text lands in the clipboard. A small model picks the image preprocessing for each capture.

**Workflow:** key → select area → text in clipboard

## Installation

Arch Linux:

```bash
yay -S sniptext
```

From source:

```bash
pip install .
```

Runtime dependencies: Tesseract, Pillow, numpy, PyYAML, loguru, pytesseract. For capture and clipboard: `slurp`, `grim` and `wl-clipboard` on Wayland, `maim` and `xclip` on X11.

Then bind a key to `sniptext --capture-now` in your compositor. See [KEYBINDINGS.md](KEYBINDINGS.md).

## Usage

```bash
sniptext                      # select a region, recognise it, copy the text
sniptext --capture-now        # the same; the form to bind to a key
sniptext --file IMAGE         # recognise an image file instead
sniptext --output FILE        # also write the text to FILE
sniptext --history [N]        # print the last N captured texts (default 10)
sniptext --profile NAME       # apply ~/.config/sniptext/profiles/NAME.yaml
sniptext --list-profiles
sniptext --print-config
```

`-c FILE` selects another config file, `-v` enables debug logging. A capture shows a desktop notification with the start of the text, or the reason when Tesseract could not run.

Exit codes: 0 on success or when no text was found, 1 when the capture, the OCR or the clipboard failed, 2 for an unreadable image file or bad arguments.

## How the pipeline is chosen

One engine, several ways to prepare the image for it. A pipeline is a preprocessing chain (inversion of dark themes, 2x upscaling, median or Gaussian filtering) plus a Tesseract page-segmentation mode. Thirteen candidates were run over a corpus of 3580 images; two were selected greedily and ship. For each capture a gradient-boosting regressor per pipeline predicts the character error rate (CER) from 12 image statistics, and the app runs the pipeline with the lowest predicted CER plus a time penalty. The model is fitted with scikit-learn offline and exported to a JSON file that the app evaluates with numpy.

On 600 held-out images (texts never seen in training), mean CER with 95% bootstrap intervals over texts:

| | CER | Time per image |
|---|---|---|
| Router (shipped) | 0.067 [0.054, 0.081] | 173 ms |
| Best single pipeline | 0.079 [0.063, 0.096] | 236 ms |
| Tesseract as version 0.4 ran it | 0.103 [0.084, 0.124] | 172 ms |

Paired differences: -0.012 [-0.024, -0.000] against the best single pipeline, -0.036 [-0.052, -0.020] against version 0.4's Tesseract.

The gain is on degraded images: 0.083 against 0.099 for the best single pipeline and 0.129 for 0.4. On clean images all three are close: 0.012, 0.009 and 0.010. On pages rendered by a browser the router reaches 0.009.

Data, method, every slice and the limitations: [docs/benchmark.md](docs/benchmark.md).

## Language support

```bash
sudo pacman -S tesseract-data-rus     # Russian
sudo pacman -S tesseract-data-ell     # Greek
sudo pacman -S tesseract-data-equ     # math symbols
```

```yaml
ocr_language: eng+rus+equ
```

Full guide: [LANGUAGES.md](LANGUAGES.md). The router was fitted and measured with `eng+rus`.

## Configuration

`~/.config/sniptext/config.yaml` is created on first run with a comment per key. `sniptext --print-config` prints the current values.

| Key | Default | Meaning |
|---|---|---|
| `ocr_language` | `eng` | Tesseract language codes, joined with `+` |
| `routing` | `true` | Choose the pipeline per image; `false` always runs the default one |
| `router_time_weight` | blank | CER traded per second; blank uses the benchmarked value, `0` ignores time |
| `max_image_size` | `4096` | Larger images are reduced to this side before OCR |
| `notification_enabled` | `true` | Desktop notification after a capture |
| `history_enabled` | `true` | Keep captured texts for `--history` |
| `history_size` | `50` | Number of texts kept |
| `display_server` | `auto` | `auto`, `wayland` or `x11` |

Images above 2 megapixels are not routed: they run the pipeline of version 0.4, which upscales only small images.

## Changes from 0.4

- EasyOCR and the merge of two engines are gone, and with them torch (about 2 GB). The benchmark report shows what that costs.
- The spelling corrector is gone: on validation texts it raised CER.
- The built-in hotkey listener, `sniptext serve`, `--client` and `--interactive` are gone. Bind `sniptext --capture-now` to a key in the compositor.
- Config keys of removed features (`ocr_engine`, `use_gpu`, `hotkey`, `adaptive_ensemble`, `enable_text_correction` and others) are ignored with a warning.
- The process now exits right after copying; before, it stayed until the clipboard changed.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The benchmark is reproduced with the five commands at the end of [docs/benchmark.md](docs/benchmark.md).
