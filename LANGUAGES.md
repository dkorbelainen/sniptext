# Language Support

SnipText uses Tesseract OCR, which supports 100+ languages.

## Installing Language Packs

```bash
sudo pacman -S tesseract-data-rus     # Russian
sudo pacman -S tesseract-data-ell     # Greek (modern)
sudo pacman -S tesseract-data-equ     # Math equations & symbols
sudo pacman -S tesseract-data-jpn     # Japanese
sudo pacman -S tesseract-data-chi-sim # Chinese Simplified
sudo pacman -S tesseract-data-fra     # French
sudo pacman -S tesseract-data-deu     # German
sudo pacman -S tesseract-data-spa     # Spanish
# See all available: pacman -Ss tesseract-data
```

## Configuration

After installing language packs, update your config file:

`~/.config/sniptext/config.yaml`

```yaml
# Single language
ocr_language: rus

# Multiple languages (combine with +)
ocr_language: eng+rus

# Scientific texts (English + Greek + Math symbols)
ocr_language: eng+ell+equ

# Check installed languages
# Run: tesseract --list-langs
```

## Language Codes

Common Tesseract language codes:
- `eng` - English
- `rus` - Russian
- `ell` - Greek (modern)
- `equ` - Mathematical equations and symbols
- `jpn` - Japanese
- `chi_sim` - Chinese Simplified
- `chi_tra` - Chinese Traditional
- `fra` - French
- `deu` - German
- `spa` - Spanish
- `ita` - Italian
- `por` - Portuguese
- `ara` - Arabic
- `hin` - Hindi
- `kor` - Korean

Full list: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
