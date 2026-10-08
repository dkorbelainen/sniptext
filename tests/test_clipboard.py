"""Tests for ClipboardManager."""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from sniptext.clipboard import ClipboardManager


def _make_manager(which_map: dict) -> ClipboardManager:
    """Return a ClipboardManager with shutil.which stubbed by which_map."""
    with patch("sniptext.clipboard.shutil.which", side_effect=lambda cmd: which_map.get(cmd)):
        return ClipboardManager()


class TestDetectClipboardTool:
    def test_prefers_wl_copy_over_xclip(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy", "xclip": "/usr/bin/xclip"})
        assert mgr.tool == "wayland"

    def test_falls_back_to_xclip(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        assert mgr.tool == "x11"
        assert "xclip" in mgr.copy_cmd[0]

    def test_falls_back_to_xsel(self):
        mgr = _make_manager({"xsel": "/usr/bin/xsel"})
        assert mgr.tool == "x11"
        assert "xsel" in mgr.copy_cmd[0]

    def test_raises_when_no_tool_found(self):
        with pytest.raises(RuntimeError, match="No clipboard tool"):
            _make_manager({})


class TestPaste:
    def test_x11_paste_returns_text(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello world"
        with patch("sniptext.clipboard.subprocess.run", return_value=mock_result):
            assert mgr.paste() == "hello world"

    def test_x11_paste_returns_none_on_failure(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("sniptext.clipboard.subprocess.run", return_value=mock_result):
            assert mgr.paste() is None

    def test_wayland_paste_uses_wl_paste(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "wayland text"
        with patch("sniptext.clipboard.subprocess.run", return_value=mock_result) as mock_run:
            result = mgr.paste()
        assert result == "wayland text"
        assert mock_run.call_args[0][0][0] == "wl-paste"

    def test_xsel_paste_uses_correct_command(self):
        mgr = _make_manager({"xsel": "/usr/bin/xsel"})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "xsel text"
        with patch("sniptext.clipboard.subprocess.run", return_value=mock_result) as mock_run:
            result = mgr.paste()
        assert result == "xsel text"
        assert "xsel" in mock_run.call_args[0][0][0]


class TestPasteEdgeCases:
    def test_unknown_tool_returns_none(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        mgr.tool = "unknown"
        assert mgr.paste() is None

    def test_exception_returns_none(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        with patch("sniptext.clipboard.subprocess.run", side_effect=OSError("oops")):
            assert mgr.paste() is None


FORKING_TOOL = """#!/bin/sh
cat > "$SNIPTEXT_TEST_CLIPBOARD"
sleep 3 &
exit 0
"""


class TestCopy:
    @pytest.mark.parametrize("tool", ["wl-copy", "xclip", "xsel"])
    def test_passes_the_text_and_succeeds_on_exit_zero(self, tool):
        mgr = _make_manager({tool: f"/usr/bin/{tool}"})
        with patch(
            "sniptext.clipboard.subprocess.run", return_value=MagicMock(returncode=0)
        ) as run:
            assert mgr.copy("héllo") is True
        assert run.call_args.args[0] == mgr.copy_cmd
        assert run.call_args.kwargs["input"] == "héllo".encode("utf-8")

    def test_output_is_not_captured(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        with patch(
            "sniptext.clipboard.subprocess.run", return_value=MagicMock(returncode=0)
        ) as run:
            mgr.copy("x")
        assert run.call_args.kwargs["stdout"] == subprocess.DEVNULL
        assert run.call_args.kwargs["stderr"] == subprocess.DEVNULL

    def test_nonzero_exit_is_a_failure(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        with patch("sniptext.clipboard.subprocess.run", return_value=MagicMock(returncode=1)):
            assert mgr.copy("x") is False

    def test_timeout_is_a_failure(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        error = subprocess.TimeoutExpired(cmd="xclip", timeout=2)
        with patch("sniptext.clipboard.subprocess.run", side_effect=error):
            assert mgr.copy("x") is False

    def test_any_other_error_is_a_failure(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        with patch("sniptext.clipboard.subprocess.run", side_effect=OSError("boom")):
            assert mgr.copy("x") is False

    def test_returns_once_a_tool_that_forks_a_server_has_exited(self, tmp_path, monkeypatch):
        # wl-copy and xclip exit at once and leave a child serving the selection.
        tool = tmp_path / "wl-copy"
        tool.write_text(FORKING_TOOL)
        tool.chmod(0o755)
        received = tmp_path / "clipboard.txt"
        monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
        monkeypatch.setenv("SNIPTEXT_TEST_CLIPBOARD", str(received))
        mgr = _make_manager({"wl-copy": str(tool)})
        start = time.monotonic()
        assert mgr.copy("héllo") is True
        assert time.monotonic() - start < 2
        assert received.read_text(encoding="utf-8") == "héllo"
