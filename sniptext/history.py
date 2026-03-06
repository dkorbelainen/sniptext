"""Capture history management for SnipText."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from loguru import logger

_DEFAULT_HISTORY_PATH = Path.home() / ".local" / "share" / "sniptext" / "history.jsonl"


class HistoryManager:
    """Append-only JSONL history of captured texts."""

    def __init__(self, path: Path = _DEFAULT_HISTORY_PATH, max_size: int = 50) -> None:
        self.path = path
        self.max_size = max_size
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        """Append a captured text entry; trims the file to max_size entries."""
        if not text:
            return
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "text": text,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._trim_locked()
        except OSError as e:
            logger.warning(f"Could not write to history file: {e}")

    def read(self, n: int = 10) -> List[dict]:
        """Return the last *n* history entries, oldest first."""
        if n <= 0:
            return []
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning(f"Could not read history file: {e}")
            return []
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed history line: {line!r}")
                continue
            if isinstance(obj, dict) and "timestamp" in obj and "text" in obj:
                entries.append(obj)
            else:
                logger.debug(f"Skipping invalid history entry (unexpected structure): {obj!r}")
        return entries[-n:]

    def _trim_locked(self) -> None:
        """Keep only the last max_size entries. Must be called with self._lock held."""
        try:
            lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            return
        if len(lines) > self.max_size:
            self.path.write_text("\n".join(lines[-self.max_size :]) + "\n", encoding="utf-8")
