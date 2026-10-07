"""Text pool for the synthetic corpus, built from pinned public sources.

Nothing fetched here is committed: sources are cached under benchmarks/data/corpus/.
"""

from __future__ import annotations

import hashlib
import random
import string
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_CACHE = Path(__file__).resolve().parent / "data" / "corpus"
_SOURCES = {
    "en.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/"
    "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/en_ewt-ud-dev.conllu",
    "ru.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-GSD/"
    "0f34b7362ac3c3facd1d6ff4b876d241bb15793e/ru_gsd-ud-dev.conllu",
    # The cleaned dev split alone yields fewer than 250 Russian prose items.
    "ru_test.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-GSD/"
    "0f34b7362ac3c3facd1d6ff4b876d241bb15793e/ru_gsd-ud-test.conllu",
}
_CODE_FILES = ("textwrap", "heapq", "bisect", "fnmatch", "shlex", "glob")
_CODE_URL = "https://raw.githubusercontent.com/python/cpython/v3.12.0/Lib/{name}.py"

_ASCII = set(string.ascii_letters + string.digits + string.punctuation + " ")
_CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

_UI = {
    "en": [
        "File", "Edit", "View", "Help", "Settings", "Open", "Save As", "Close", "Undo", "Redo",
        "Copy", "Paste", "Search", "Preferences", "New Project", "Sign in", "Log out", "Username",
        "Password", "Remember me", "Dark mode", "Notifications", "Cancel", "Apply", "Downloads",
        "History", "Print", "Export", "Import", "Zoom in",
    ],
    "ru": [
        "Файл", "Правка", "Вид", "Справка", "Настройки", "Открыть", "Сохранить как", "Закрыть",
        "Отменить", "Повторить", "Копировать", "Вставить", "Поиск", "Параметры", "Новый проект",
        "Войти", "Выйти", "Имя пользователя", "Пароль", "Запомнить меня", "Тёмная тема",
        "Уведомления", "Отмена", "Применить", "Загрузки", "История", "Печать", "Экспорт",
        "Импорт", "Увеличить",
    ],
}  # fmt: skip


@dataclass(frozen=True)
class TextItem:
    text_id: str
    lang: str
    content: str
    text: str


def _item(lang: str, content: str, text: str) -> TextItem:
    digest = hashlib.sha1(f"{lang}|{content}|{text}".encode("utf-8")).hexdigest()[:12]
    return TextItem(digest, lang, content, text)


def _fetch(name: str, url: str) -> str:
    path = _CACHE / name
    if not path.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


def parse_conllu_sentences(raw: str) -> list[str]:
    """Sentence texts from the '# text = ...' comment lines of a CoNLL-U file."""
    prefix = "# text = "
    return [line[len(prefix) :].strip() for line in raw.splitlines() if line.startswith(prefix)]


def clean_sentences(sentences: list[str], lang: str) -> list[str]:
    """Keep sentences a screen font can render and an OCR engine can be scored on."""
    allowed = _ASCII | (_CYRILLIC if lang == "ru" else set())
    out: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        words = sentence.split()
        if not 3 <= len(words) <= 22 or len(sentence) > 140:
            continue
        if "&" in sentence or not set(sentence) <= allowed:
            continue
        has_cyrillic = bool(set(sentence) & _CYRILLIC)
        if has_cyrillic != (lang == "ru") or sentence in seen:
            continue
        seen.add(sentence)
        out.append(sentence)
    return out


def parse_code_lines(raw: str) -> list[str]:
    """Code lines short enough to render on one line, without comments or docstrings."""
    out = []
    for line in raw.splitlines():
        line = line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or '"""' in line or "'''" in line:
            continue
        if len(line) > 64 or not set(line) <= _ASCII:
            continue
        out.append(line)
    return out


def _prose_items(sentences: list[str], lang: str, n: int, rng: random.Random) -> list[TextItem]:
    pool = list(sentences)
    rng.shuffle(pool)
    items: list[TextItem] = []
    position = 0
    while len(items) < n and position < len(pool):
        count = rng.randint(1, 3)
        chunk = pool[position : position + count]
        position += count
        lines = textwrap.wrap(" ".join(chunk), width=rng.choice([40, 52, 64]))[:8]
        items.append(_item(lang, "prose", "\n".join(lines)))
    return items


def _code_items(lines: list[str], n: int, rng: random.Random) -> list[TextItem]:
    chunks: list[str] = []
    position = 0
    while position < len(lines):
        count = rng.randint(2, 6)
        chunk = lines[position : position + count]
        position += count
        if len(chunk) >= 2:
            chunks.append("\n".join(chunk))
    rng.shuffle(chunks)
    return [_item("en", "code", chunk) for chunk in chunks[:n]]


def _ui_items(lang: str, n: int, rng: random.Random) -> list[TextItem]:
    items: dict[str, TextItem] = {}
    while len(items) < n:
        labels = rng.sample(_UI[lang], rng.randint(3, 6))
        if rng.random() < 0.3:
            labels = [f"{label}    Ctrl+{rng.choice(string.ascii_uppercase)}" for label in labels]
        if rng.random() < 0.3:
            text = "  ".join(labels[:4])
        else:
            text = "\n".join(labels)
        item = _item(lang, "ui", text)
        items[item.text_id] = item
    return list(items.values())


def build_items(en, ru, code, seed=42, n_prose=250, n_code=150, n_ui=50) -> list[TextItem]:
    """Deterministic text pool. No sentence or code line is used by two items."""
    rng = random.Random(seed)
    items = (
        _prose_items(en, "en", n_prose, rng)
        + _prose_items(ru, "ru", n_prose, rng)
        + _code_items(code, n_code, rng)
        + (_ui_items("en", n_ui, rng) if n_ui else [])
        + (_ui_items("ru", n_ui, rng) if n_ui else [])
    )
    unique: dict[str, TextItem] = {}
    for item in items:
        unique.setdefault(item.text_id, item)
    return list(unique.values())


def load_items(seed: int = 42) -> list[TextItem]:
    """Fetch the pinned sources (cached) and build the pool."""
    en = clean_sentences(parse_conllu_sentences(_fetch("en.conllu", _SOURCES["en.conllu"])), "en")
    ru_raw: list[str] = []
    for name in ("ru.conllu", "ru_test.conllu"):
        ru_raw += parse_conllu_sentences(_fetch(name, _SOURCES[name]))
    ru = clean_sentences(ru_raw, "ru")
    code: list[str] = []
    for name in _CODE_FILES:
        code += parse_code_lines(_fetch(f"{name}.py", _CODE_URL.format(name=name)))
    return build_items(en, ru, code, seed=seed)


if __name__ == "__main__":
    pool = load_items()
    counts: dict[tuple, int] = {}
    for it in pool:
        counts[(it.lang, it.content)] = counts.get((it.lang, it.content), 0) + 1
    print(len(pool), counts)
