"""Tests for Config."""

from pathlib import Path

from sniptext.config import Config


class TestConfigDefaults:
    def test_hotkey(self):
        assert Config().hotkey == "<ctrl>+<alt>+t"

    def test_ocr_language(self):
        assert Config().ocr_language == "eng"

    def test_ocr_engine(self):
        assert Config().ocr_engine == "ensemble"

    def test_model_path_set(self):
        config = Config()
        assert config.ocr_model_path is not None
        assert isinstance(config.ocr_model_path, Path)

    def test_text_correction_enabled_by_default(self):
        assert Config().enable_text_correction is True

    def test_adaptive_ensemble_enabled_by_default(self):
        assert Config().adaptive_ensemble is True


class TestConfigSaveLoad:
    def test_round_trip(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        original = Config()
        original.save(config_path)

        loaded = Config.load(config_path)

        assert loaded.hotkey == original.hotkey
        assert loaded.ocr_language == original.ocr_language
        assert loaded.ocr_engine == original.ocr_engine
        assert loaded.enable_text_correction == original.enable_text_correction

    def test_load_creates_file_if_missing(self, tmp_path):
        config_path = tmp_path / "new_config.yaml"
        assert not config_path.exists()

        config = Config.load(config_path)

        assert config_path.exists()
        assert config.hotkey == "<ctrl>+<alt>+t"

    def test_save_load_custom_values(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        original = Config(hotkey="<ctrl>+<shift>+s", ocr_language="rus", ocr_engine="tesseract")
        original.save(config_path)

        loaded = Config.load(config_path)

        assert loaded.hotkey == "<ctrl>+<shift>+s"
        assert loaded.ocr_language == "rus"
        assert loaded.ocr_engine == "tesseract"

    def test_load_ignores_deprecated_keys(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        # Write a config with deprecated keys manually
        config_path.write_text(
            "hotkey: <ctrl>+<alt>+t\n"
            "ocr_language: eng\n"
            "preprocessing_enabled: true\n"  # deprecated
            "save_history: false\n"  # deprecated
        )
        # Should not raise
        config = Config.load(config_path)
        assert config.ocr_language == "eng"

    def test_load_ignores_unknown_keys(self, tmp_path):
        """Config.load must not crash on unknown YAML keys."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "hotkey: <ctrl>+<alt>+t\nocr_language: eng\ntotally_unknown_key: some_value\n"
        )
        config = Config.load(config_path)
        assert config.hotkey == "<ctrl>+<alt>+t"
        assert not hasattr(config, "totally_unknown_key")
