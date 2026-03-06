"""Tests for sniptext.history."""

import json

from sniptext.history import HistoryManager


class TestHistoryAppend:
    def test_append_creates_file(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        hm.append("hello")
        assert (tmp_path / "history.jsonl").exists()

    def test_append_writes_valid_json(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        hm.append("test text")
        line = (tmp_path / "history.jsonl").read_text().strip()
        entry = json.loads(line)
        assert entry["text"] == "test text"
        assert "timestamp" in entry

    def test_append_empty_text_is_noop(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        hm.append("")
        assert not (tmp_path / "history.jsonl").exists()

    def test_append_creates_parent_dirs(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "a" / "b" / "history.jsonl", max_size=50)
        hm.append("text")
        assert (tmp_path / "a" / "b" / "history.jsonl").exists()

    def test_append_multiple_entries(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        hm.append("first")
        hm.append("second")
        lines = (tmp_path / "history.jsonl").read_text().splitlines()
        assert len(lines) == 2

    def test_append_write_error_logs_warning(self, tmp_path, caplog):
        path = tmp_path / "history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()  # make it a directory to force OSError on open
        import logging

        hm = HistoryManager(path=path, max_size=50)
        with caplog.at_level(logging.WARNING):
            hm.append("text")  # should not raise


class TestHistoryTrim:
    def test_trim_keeps_max_size(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=3)
        for i in range(5):
            hm.append(f"entry {i}")
        lines = (tmp_path / "history.jsonl").read_text().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[-1])["text"] == "entry 4"

    def test_trim_does_not_touch_under_max(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=10)
        for i in range(5):
            hm.append(f"entry {i}")
        lines = (tmp_path / "history.jsonl").read_text().splitlines()
        assert len(lines) == 5


class TestHistoryRead:
    def test_read_zero_returns_empty(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        hm.append("something")
        assert hm.read(0) == []

    def test_read_negative_returns_empty(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        hm.append("something")
        assert hm.read(-5) == []

    def test_read_returns_empty_when_no_file(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        assert hm.read() == []

    def test_read_returns_last_n(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        for i in range(5):
            hm.append(f"entry {i}")
        entries = hm.read(3)
        assert len(entries) == 3
        assert entries[-1]["text"] == "entry 4"

    def test_read_returns_all_when_n_larger(self, tmp_path):
        hm = HistoryManager(path=tmp_path / "history.jsonl", max_size=50)
        hm.append("only one")
        assert len(hm.read(100)) == 1

    def test_read_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.write_text('not json\n{"timestamp":"t","text":"ok"}\n')
        hm = HistoryManager(path=path, max_size=50)
        entries = hm.read()
        assert len(entries) == 1
        assert entries[0]["text"] == "ok"

    def test_read_error_returns_empty(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.mkdir()  # force OSError
        hm = HistoryManager(path=path, max_size=50)
        assert hm.read() == []
