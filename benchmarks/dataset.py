"""Download and parse the SROIE (ICDAR-2019) receipt OCR dataset."""

import subprocess
from pathlib import Path
from typing import Iterator, Optional, Tuple

_REPO_URL = "https://github.com/zzzDavid/ICDAR-2019-SROIE"
_DATA_DIR = Path(__file__).resolve().parent / "data" / "ICDAR-2019-SROIE"


def ensure_dataset() -> Path:
    """Clone the SROIE mirror into the local cache if absent. Returns data root."""
    if not _DATA_DIR.exists():
        _DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, str(_DATA_DIR)],
            check=True,
        )
    return _DATA_DIR / "data"


def _parse_box_file(box_path: Path) -> str:
    """Reconstruct full transcription from a SROIE box annotation file."""
    lines = []
    for raw in box_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # The transcription may itself contain commas; split off the 8 coords.
        parts = raw.split(",", 8)
        if len(parts) < 9:
            continue
        lines.append(parts[8])
    return "\n".join(lines)


def load_sroie(limit: Optional[int] = None) -> Iterator[Tuple[Path, str]]:
    """Yield (image_path, ground_truth_text) for each SROIE sample."""
    root = ensure_dataset()
    img_dir = root / "img"
    box_dir = root / "box"

    count = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        box_path = box_dir / (img_path.stem + ".csv")
        if not box_path.exists():
            continue
        gt = _parse_box_file(box_path)
        if not gt.strip():
            continue
        yield img_path, gt
        count += 1
        if limit is not None and count >= limit:
            return
