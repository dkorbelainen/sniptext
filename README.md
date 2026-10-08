# SnipText

[![CI](https://github.com/dkorbelainen/sniptext/actions/workflows/ci.yml/badge.svg)](https://github.com/dkorbelainen/sniptext/actions/workflows/ci.yml)

Select a screen region; Tesseract reads it and the text lands in the clipboard.

## Install

```bash
yay -S sniptext        # Arch Linux
pip install .          # from source
```

Needs Tesseract, plus `slurp`, `grim` and `wl-clipboard` on Wayland or `maim` and `xclip` on X11. Bind a key to `sniptext --capture-now`: [KEYBINDINGS.md](KEYBINDINGS.md).

## Usage

```bash
sniptext                      # select a region, recognise it, copy the text
sniptext --file IMAGE         # recognise an image file instead
sniptext --output FILE        # also write the text to FILE
sniptext --history [N]        # print the last N captured texts (default 10)
sniptext --profile NAME       # apply ~/.config/sniptext/profiles/NAME.yaml
sniptext --list-profiles
sniptext --print-config
```

`-c FILE` selects another config file, `-v` enables debug logging.

Exit codes: 0 on success or when no text was found, 1 when the capture, the OCR or the clipboard failed, 2 for an unreadable image file or bad arguments.

## How it works

Every capture is read with a default preprocessing pipeline. From 12 image statistics and the word confidences of that pass, a gradient-boosting model predicts the character error rate of a second pipeline, which runs only when it is expected to pay for its time. The two pipelines were selected from 13 candidates. The model is fitted with scikit-learn and ships as a JSON file evaluated with numpy.

Benchmark: [docs/benchmark.md](docs/benchmark.md).

## Configuration

`~/.config/sniptext/config.yaml` is created on first run.

| Key | Default | Meaning |
|---|---|---|
| `ocr_language` | `eng` | Tesseract language codes, joined with `+` |
| `routing` | `true` | Choose the pipeline per image; `false` runs only the default one |
| `router_time_weight` | blank | CER traded per second; blank uses the benchmarked value, `0` ignores time |
| `max_image_size` | `4096` | Larger images are reduced to this side before OCR |
| `notification_enabled` | `true` | Desktop notification after a capture |
| `history_enabled` | `true` | Keep captured texts for `--history` |
| `history_size` | `50` | Number of texts kept |
| `display_server` | `auto` | `auto`, `wayland` or `x11` |

Images above 2 megapixels always run the preprocessing of version 0.4.

Other languages: [LANGUAGES.md](LANGUAGES.md). The router was fitted and measured with `eng+rus`.
