import subprocess
from unittest.mock import MagicMock

from sniptext import notify


def test_preview_shortens_and_flattens():
    assert notify.preview("short") == "short"
    assert notify.preview("a\nb") == "a b"
    long = "x" * 80
    assert notify.preview(long) == "x" * 50 + "…"


def test_send_calls_notify_send(monkeypatch):
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(subprocess, "run", run)
    notify.send("hello")
    assert run.call_args.args[0] == ["notify-send", "SnipText", "hello"]


def test_send_survives_a_missing_binary(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError()))
    notify.send("hello")


def test_send_survives_a_timeout(monkeypatch):
    error = subprocess.TimeoutExpired("notify-send", 2)
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=error))
    notify.send("hello")
