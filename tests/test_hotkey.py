"""Tests for HotkeyManager logic (no real keyboard listener started)."""

from unittest.mock import MagicMock, patch

from sniptext.config import Config
from sniptext.hotkey import HotkeyManager


def _make_manager(hotkey: str = "<ctrl>+<alt>+t") -> HotkeyManager:
    """Return a HotkeyManager with heavyweight deps mocked out."""
    return HotkeyManager(
        config=Config(hotkey=hotkey),
        screen_capture=MagicMock(),
        ocr_engine=MagicMock(),
        clipboard_manager=MagicMock(),
    )


class TestParseHotkey:
    def test_parses_key(self):
        mgr = _make_manager("<ctrl>+<alt>+t")
        assert mgr.key == "t"

    def test_parses_ctrl_modifier(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<alt>+t")
        assert keyboard.Key.ctrl_l in mgr.modifiers
        assert keyboard.Key.ctrl_r in mgr.modifiers

    def test_parses_alt_modifier(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<alt>+t")
        assert keyboard.Key.alt_l in mgr.modifiers
        assert keyboard.Key.alt_r in mgr.modifiers

    def test_shift_modifier(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<shift>+s")
        assert keyboard.Key.shift_l in mgr.modifiers
        assert mgr.key == "s"

    def test_super_modifier(self):
        from pynput import keyboard

        mgr = _make_manager("<super>+s")
        assert keyboard.Key.cmd in mgr.modifiers
        assert mgr.key == "s"

    def test_no_modifier_logs_warning(self):
        with patch("sniptext.hotkey.logger.warning") as mock_warn:
            _make_manager("t")
        # At least one warning must mention the missing modifier
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("modifier" in m.lower() for m in messages)

    def test_with_modifier_no_warning(self):
        with patch("sniptext.hotkey.logger.warning") as mock_warn:
            _make_manager("<ctrl>+<alt>+t")
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert not any("modifier" in m.lower() for m in messages)


class TestIsHotkeyPressed:
    def test_full_combo_returns_true(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<alt>+t")
        keys = {keyboard.Key.ctrl_l, keyboard.Key.alt_l, "t"}
        assert mgr._is_hotkey_pressed(keys) is True

    def test_missing_modifier_returns_false(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<alt>+t")
        keys = {keyboard.Key.ctrl_l, "t"}  # no alt
        assert mgr._is_hotkey_pressed(keys) is False

    def test_missing_key_returns_false(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<alt>+t")
        keys = {keyboard.Key.ctrl_l, keyboard.Key.alt_l}  # no 't'
        assert mgr._is_hotkey_pressed(keys) is False

    def test_right_side_modifiers_accepted(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<alt>+t")
        keys = {keyboard.Key.ctrl_r, keyboard.Key.alt_r, "t"}
        assert mgr._is_hotkey_pressed(keys) is True

    def test_extra_keys_still_match(self):
        from pynput import keyboard

        mgr = _make_manager("<ctrl>+<alt>+t")
        keys = {keyboard.Key.ctrl_l, keyboard.Key.alt_l, "t", keyboard.Key.shift_l}
        assert mgr._is_hotkey_pressed(keys) is True

    def test_super_required_when_configured(self):
        mgr = _make_manager("<super>+s")
        keys_without_super = {"s"}
        assert mgr._is_hotkey_pressed(keys_without_super) is False

    def test_super_combo_matches_when_pressed(self):
        from pynput import keyboard

        mgr = _make_manager("<super>+s")
        keys = {keyboard.Key.cmd, "s"}
        assert mgr._is_hotkey_pressed(keys) is True

    def test_no_modifier_hotkey_matches_on_key_alone(self):
        """When modifiers is empty, any press of the key triggers (modifiers_ok=True)."""
        mgr = _make_manager("t")  # no modifier — this is intentionally warned about
        assert mgr._is_hotkey_pressed({"t"}) is True

    def test_no_modifier_hotkey_does_not_match_wrong_key(self):
        mgr = _make_manager("t")
        assert mgr._is_hotkey_pressed({"s"}) is False


class TestOnHotkeyTriggered:
    def test_calls_capture_ocr_clipboard_in_order(self):
        mgr = _make_manager()
        mgr.screen_capture.capture_region.return_value = MagicMock()
        mgr.ocr_engine.recognize.return_value = "some text"
        mgr.clipboard_manager.copy.return_value = True

        with patch.object(mgr, "_show_notification"):
            mgr._on_hotkey_triggered()

        mgr.screen_capture.capture_region.assert_called_once()
        mgr.ocr_engine.recognize.assert_called_once()
        mgr.clipboard_manager.copy.assert_called_once_with("some text")

    def test_skips_clipboard_when_no_text(self):
        mgr = _make_manager()
        mgr.screen_capture.capture_region.return_value = MagicMock()
        mgr.ocr_engine.recognize.return_value = ""

        mgr._on_hotkey_triggered()

        mgr.clipboard_manager.copy.assert_not_called()

    def test_skips_everything_when_capture_fails(self):
        mgr = _make_manager()
        mgr.screen_capture.capture_region.return_value = None

        mgr._on_hotkey_triggered()

        mgr.ocr_engine.recognize.assert_not_called()
        mgr.clipboard_manager.copy.assert_not_called()

    def test_processing_flag_cleared_after_run(self):
        mgr = _make_manager()
        mgr.screen_capture.capture_region.return_value = MagicMock()
        mgr.ocr_engine.recognize.return_value = "text"
        mgr.clipboard_manager.copy.return_value = True

        with patch.object(mgr, "_show_notification"):
            mgr._on_hotkey_triggered()

        assert not mgr._processing.is_set()

    def test_processing_flag_cleared_even_on_exception(self):
        mgr = _make_manager()
        mgr.screen_capture.capture_region.side_effect = RuntimeError("capture error")

        mgr._on_hotkey_triggered()

        assert not mgr._processing.is_set()

    def test_clipboard_failure_does_not_raise(self):
        mgr = _make_manager()
        mgr.screen_capture.capture_region.return_value = MagicMock()
        mgr.ocr_engine.recognize.return_value = "some text"
        mgr.clipboard_manager.copy.return_value = False

        mgr._on_hotkey_triggered()  # must not raise

        mgr.clipboard_manager.copy.assert_called_once_with("some text")

    def test_notification_sent_on_success(self):
        mgr = _make_manager()
        mgr.screen_capture.capture_region.return_value = MagicMock()
        mgr.ocr_engine.recognize.return_value = "hello"
        mgr.clipboard_manager.copy.return_value = True
        mgr.config.notification_enabled = True

        with patch.object(mgr, "_show_notification") as mock_notify:
            mgr._on_hotkey_triggered()

        mock_notify.assert_called_once()


class TestShowNotification:
    def test_calls_notify_send(self):
        mgr = _make_manager()
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("sniptext.hotkey.subprocess.run", return_value=mock_result) as mock_run:
            mgr._show_notification("hello")
        assert mock_run.called
        assert mock_run.call_args[0][0][:2] == ["notify-send", "SnipText"]

    def test_handles_notify_send_not_found(self):
        mgr = _make_manager()
        with patch("sniptext.hotkey.subprocess.run", side_effect=FileNotFoundError):
            mgr._show_notification("hello")  # must not raise

    def test_notification_disabled_not_called(self):
        mgr = HotkeyManager(
            config=Config(notification_enabled=False),
            screen_capture=MagicMock(),
            ocr_engine=MagicMock(),
            clipboard_manager=MagicMock(),
        )
        mgr.screen_capture.capture_region.return_value = MagicMock()
        mgr.ocr_engine.recognize.return_value = "text"
        mgr.clipboard_manager.copy.return_value = True

        with patch.object(mgr, "_show_notification") as mock_notify:
            mgr._on_hotkey_triggered()

        mock_notify.assert_not_called()

    def test_nonzero_returncode_does_not_raise(self):
        mgr = _make_manager()
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("sniptext.hotkey.subprocess.run", return_value=mock_result):
            mgr._show_notification("hello")  # must not raise

    def test_unexpected_exception_does_not_raise(self):
        mgr = _make_manager()
        with patch("sniptext.hotkey.subprocess.run", side_effect=RuntimeError("boom")):
            mgr._show_notification("hello")  # must not raise


class TestStop:
    def test_stop_calls_listener_stop(self):
        mgr = _make_manager()
        mgr.listener = MagicMock()
        mgr.stop()
        mgr.listener.stop.assert_called_once()

    def test_stop_is_noop_when_no_listener(self):
        mgr = _make_manager()
        mgr.listener = None
        mgr.stop()  # must not raise


class TestWaylandDetection:
    def test_wayland_detected_via_wayland_display(self):
        mgr = _make_manager()
        with patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"}):
            with patch("sniptext.hotkey.logger.warning") as mock_warning:
                with patch("pynput.keyboard.Listener") as mock_listener:
                    listener_instance = MagicMock()
                    listener_instance.join.side_effect = KeyboardInterrupt
                    mock_listener.return_value.__enter__ = MagicMock(return_value=listener_instance)
                    mock_listener.return_value.__exit__ = MagicMock(return_value=False)
                    try:
                        mgr.start()
                    except KeyboardInterrupt:
                        pass
        mock_listener.assert_called()
        mock_warning.assert_called()

    def test_no_warning_without_wayland_display(self):
        mgr = _make_manager()
        env = {k: v for k, v in __import__("os").environ.items() if k != "WAYLAND_DISPLAY"}
        with patch.dict("os.environ", env, clear=True):
            with patch("sniptext.hotkey.logger.warning") as mock_warning:
                with patch("pynput.keyboard.Listener") as mock_listener:
                    listener_instance = MagicMock()
                    listener_instance.join.side_effect = KeyboardInterrupt
                    mock_listener.return_value.__enter__ = MagicMock(return_value=listener_instance)
                    mock_listener.return_value.__exit__ = MagicMock(return_value=False)
                    try:
                        mgr.start()
                    except KeyboardInterrupt:
                        pass
        mock_listener.assert_called()
        mock_warning.assert_not_called()
