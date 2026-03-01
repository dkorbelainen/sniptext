"""Tests for ScreenCapture."""

import os
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from sniptext.capture import ScreenCapture
from sniptext.config import Config


def _make_capture(which_map: dict, config: Config | None = None, env: dict | None = None):
    """Return a ScreenCapture with shutil.which and os.environ stubbed."""
    cfg = config or Config()
    env = env or {}
    with patch("sniptext.capture.shutil.which", side_effect=lambda cmd: which_map.get(cmd)):
        with patch.dict(os.environ, env, clear=True):
            return ScreenCapture(cfg)


class TestDetectDisplayServer:
    def test_explicit_wayland_config(self):
        cap = _make_capture(
            {"slurp": "/usr/bin/slurp", "grim": "/usr/bin/grim"},
            config=Config(display_server="wayland"),
        )
        assert cap.display_server == "wayland"

    def test_explicit_x11_config(self):
        cap = _make_capture({"maim": "/usr/bin/maim"}, config=Config(display_server="x11"))
        assert cap.display_server == "x11"

    def test_auto_detects_wayland_from_env(self):
        cap = _make_capture(
            {"slurp": "/usr/bin/slurp", "grim": "/usr/bin/grim"},
            env={"WAYLAND_DISPLAY": "wayland-0"},
        )
        assert cap.display_server == "wayland"

    def test_auto_detects_x11_without_env(self):
        cap = _make_capture({"maim": "/usr/bin/maim"})
        assert cap.display_server == "x11"


class TestDetectCaptureTools:
    def test_wayland_prefers_slurp_grim(self):
        cap = _make_capture(
            {"slurp": "/usr/bin/slurp", "grim": "/usr/bin/grim", "grimshot": "/usr/bin/grimshot"},
            config=Config(display_server="wayland"),
        )
        assert cap.capture_method == "slurp_grim"

    def test_wayland_falls_back_to_grimshot(self):
        cap = _make_capture(
            {"grimshot": "/usr/bin/grimshot"},
            config=Config(display_server="wayland"),
        )
        assert cap.capture_method == "grimshot"

    def test_wayland_raises_when_no_tool(self):
        with pytest.raises(RuntimeError, match="Wayland screenshot"):
            _make_capture({}, config=Config(display_server="wayland"))

    def test_x11_prefers_maim(self):
        cap = _make_capture(
            {"maim": "/usr/bin/maim", "scrot": "/usr/bin/scrot"},
            config=Config(display_server="x11"),
        )
        assert cap.capture_method == "maim"

    def test_x11_falls_back_to_scrot(self):
        cap = _make_capture({"scrot": "/usr/bin/scrot"}, config=Config(display_server="x11"))
        assert cap.capture_method == "scrot"

    def test_x11_falls_back_to_import(self):
        cap = _make_capture({"import": "/usr/bin/import"}, config=Config(display_server="x11"))
        assert cap.capture_method == "import"

    def test_x11_raises_when_no_tool(self):
        with pytest.raises(RuntimeError, match="X11 screenshot"):
            _make_capture({}, config=Config(display_server="x11"))


class TestCaptureRegion:
    def test_returns_none_when_capture_fails(self):
        cap = _make_capture({"maim": "/usr/bin/maim"}, config=Config(display_server="x11"))
        with patch.object(cap, "_capture_to_file", return_value=False):
            assert cap.capture_region() is None

    def test_returns_array_on_success(self, tmp_path):
        cap = _make_capture({"maim": "/usr/bin/maim"}, config=Config(display_server="x11"))
        img_path = tmp_path / "test.png"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(img_path)

        def fake_capture(path):
            import shutil

            shutil.copy(img_path, path)
            return True

        with patch.object(cap, "_capture_to_file", side_effect=fake_capture):
            result = cap.capture_region()
        assert isinstance(result, np.ndarray)
        assert result.shape[:2] == (10, 10)
