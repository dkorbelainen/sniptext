# Languages

Install the Tesseract data and list the codes in the config, joined with `+`.

```bash
sudo pacman -S tesseract-data-rus     # Russian
sudo pacman -S tesseract-data-ell     # Greek
sudo pacman -S tesseract-data-equ     # math symbols
pacman -Ss tesseract-data             # everything available
tesseract --list-langs                # what is installed
```

`~/.config/sniptext/config.yaml`:

```yaml
ocr_language: eng+rus
```

Language codes: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html

A language whose data is not installed is reported as an error. The router was fitted and measured with `eng+rus`.
