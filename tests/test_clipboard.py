"""Tests for ClipboardManager."""

import subprocess
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


class TestCopy:
    def test_x11_copy_returns_true_on_success(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")
        with patch("sniptext.clipboard.subprocess.Popen", return_value=mock_proc):
            assert mgr.copy("hello") is True

    def test_x11_copy_returns_false_on_nonzero_exit(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"error")
        with patch("sniptext.clipboard.subprocess.Popen", return_value=mock_proc):
            assert mgr.copy("hello") is False

    def test_x11_copy_returns_false_on_timeout(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="xclip", timeout=2)
        with patch("sniptext.clipboard.subprocess.Popen", return_value=mock_proc):
            assert mgr.copy("hello") is False

    def test_wayland_copy_returns_true_on_success(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running = success
        mock_proc.stdin = MagicMock()
        with patch("sniptext.clipboard.subprocess.Popen", return_value=mock_proc):
            with patch("sniptext.clipboard.time.sleep"):
                assert mgr.copy("hello") is True
        assert mgr._wl_process is mock_proc

    def test_wayland_copy_kills_previous_process_on_next_copy(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        first_proc = MagicMock()
        first_proc.poll.return_value = None  # still running
        first_proc.stdin = MagicMock()
        second_proc = MagicMock()
        second_proc.poll.return_value = None
        second_proc.stdin = MagicMock()
        with patch("sniptext.clipboard.time.sleep"):
            with patch(
                "sniptext.clipboard.subprocess.Popen", side_effect=[first_proc, second_proc]
            ):
                mgr.copy("first")
                mgr.copy("second")
        first_proc.terminate.assert_called_once()
        assert mgr._wl_process is second_proc

    def test_wayland_copy_broken_pipe_returns_false(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.stdin.write.side_effect = BrokenPipeError
        mock_proc.stderr.read.return_value = b"gone"
        with patch("sniptext.clipboard.subprocess.Popen", return_value=mock_proc):
            assert mgr.copy("hello") is False

    def test_wayland_copy_returns_false_when_process_exits_with_error(self):
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_proc.stdin = MagicMock()
        mock_proc.stderr.read.return_value = b"compositor error"
        with patch("sniptext.clipboard.subprocess.Popen", return_value=mock_proc):
            with patch("sniptext.clipboard.time.sleep"):
                assert mgr.copy("hello") is False

    def test_wayland_copy_returns_false_when_process_exits_with_zero(self):
        # Any early exit (even rc=0) means the selection won't be served.
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdin = MagicMock()
        mock_proc.stderr.read.return_value = b""
        with patch("sniptext.clipboard.subprocess.Popen", return_value=mock_proc):
            with patch("sniptext.clipboard.time.sleep"):
                assert mgr.copy("hello") is False


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


class TestClipboardCleanup:
    def test_cleanup_terminates_wl_process(self):
        """cleanup() must terminate a running wl-copy process."""
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mgr._wl_process = mock_proc

        mgr.cleanup()

        mock_proc.terminate.assert_called_once()
        assert mgr._wl_process is None

    def test_cleanup_is_noop_when_no_process(self):
        """cleanup() must not raise if no wl-copy process is running."""
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mgr._wl_process = None
        mgr.cleanup()  # must not raise

    def test_cleanup_skips_already_exited_process(self):
        """cleanup() must not call terminate() if the process already exited."""
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        mgr._wl_process = mock_proc

        mgr.cleanup()

        mock_proc.terminate.assert_not_called()


class TestWlCopyTerminateTimeout:
    """Cover the kill() fallback when terminate() leaves the process running."""

    def _make_wayland_manager(self):
        return _make_manager({"wl-copy": "/usr/bin/wl-copy"})

    def test_kill_called_when_terminate_times_out(self):
        """If wl-copy ignores terminate(), kill() must be called."""
        mgr = self._make_wayland_manager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process still running
        mock_proc.terminate.return_value = None
        # First wait() (after terminate) times out; second (after kill) succeeds
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("wl-copy", 1.0), None]
        mgr._wl_process = mock_proc

        new_proc = MagicMock()
        new_proc.poll.return_value = None
        new_proc.stdin = MagicMock()

        with patch("sniptext.clipboard.subprocess.Popen", return_value=new_proc):
            with patch("sniptext.clipboard.time.sleep"):
                mgr.copy("hello")

        mock_proc.kill.assert_called_once()
        assert mgr._wl_process is new_proc

    def test_wl_process_reset_after_terminate_exception(self):
        """If terminate() itself raises, _wl_process must still be reset."""
        mgr = self._make_wayland_manager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate.side_effect = OSError("gone")
        mgr._wl_process = mock_proc

        new_proc = MagicMock()
        new_proc.poll.return_value = None
        new_proc.stdin = MagicMock()

        with patch("sniptext.clipboard.subprocess.Popen", return_value=new_proc):
            with patch("sniptext.clipboard.time.sleep"):
                mgr.copy("hello")

        assert mgr._wl_process is new_proc

    def test_kill_wait_timeout_does_not_raise(self):
        """If kill() wait also times out, copy() must still succeed (best-effort)."""
        mgr = self._make_wayland_manager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("wl-copy", 1.0)
        mgr._wl_process = mock_proc

        new_proc = MagicMock()
        new_proc.poll.return_value = None
        new_proc.stdin = MagicMock()

        with patch("sniptext.clipboard.subprocess.Popen", return_value=new_proc):
            with patch("sniptext.clipboard.time.sleep"):
                result = mgr.copy("hello")

        assert result is True
        assert mgr._wl_process is new_proc


class TestCopyGenericException:
    def test_generic_exception_in_copy_returns_false(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        with patch("sniptext.clipboard.subprocess.Popen", side_effect=OSError("no xclip")):
            assert mgr.copy("hello") is False


class TestCleanupKillPath:
    def test_cleanup_calls_kill_when_terminate_times_out(self):
        """cleanup() must call kill() if terminate() leaves the process running."""
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("wl-copy", 1.0),
            None,
        ]
        mgr._wl_process = mock_proc

        mgr.cleanup()

        mock_proc.kill.assert_called_once()
        assert mgr._wl_process is None

    def test_cleanup_kill_wait_timeout_does_not_raise(self):
        """If the kill() wait also times out, cleanup() must still complete cleanly."""
        mgr = _make_manager({"wl-copy": "/usr/bin/wl-copy"})
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("wl-copy", 1.0)
        mgr._wl_process = mock_proc

        mgr.cleanup()  # must not raise

        assert mgr._wl_process is None


class TestPasteEdgeCases:
    def test_unknown_tool_returns_none(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        mgr.tool = "unknown"
        assert mgr.paste() is None

    def test_exception_returns_none(self):
        mgr = _make_manager({"xclip": "/usr/bin/xclip"})
        with patch("sniptext.clipboard.subprocess.run", side_effect=OSError("oops")):
            assert mgr.paste() is None
