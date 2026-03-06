"""Tests for Config."""

from pathlib import Path

import pytest

from sniptext.config import Config


class TestConfigDefaults:
    def test_hotkey(self):
        assert Config().hotkey == "<ctrl>+<alt>+t"

    def test_max_image_size_default(self):
        assert Config().max_image_size == 4096

    def test_use_gpu_default(self):
        assert Config().use_gpu is True

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


class TestConfigValidation:
    def test_invalid_ocr_engine_resets_to_ensemble(self):
        config = Config(ocr_engine="foobar")
        assert config.ocr_engine == "ensemble"

    def test_valid_ocr_engine_accepted(self):
        for engine in ("ensemble", "tesseract", "easyocr"):
            assert Config(ocr_engine=engine).ocr_engine == engine

    def test_invalid_display_server_resets_to_auto(self):
        config = Config(display_server="foobar")
        assert config.display_server == "auto"

    def test_valid_display_server_accepted(self):
        for ds in ("auto", "wayland", "x11"):
            assert Config(display_server=ds).display_server == ds

    def test_zero_confidence_threshold_resets(self):
        config = Config(ocr_confidence_threshold=0.0)
        assert config.ocr_confidence_threshold == 0.6

    def test_negative_confidence_threshold_resets(self):
        config = Config(ocr_confidence_threshold=-0.5)
        assert config.ocr_confidence_threshold == 0.6

    def test_confidence_threshold_above_one_resets(self):
        config = Config(ocr_confidence_threshold=1.5)
        assert config.ocr_confidence_threshold == 0.6

    def test_valid_confidence_threshold_accepted(self):
        assert Config(ocr_confidence_threshold=1.0).ocr_confidence_threshold == 1.0
        assert Config(ocr_confidence_threshold=0.01).ocr_confidence_threshold == 0.01

    def test_max_image_size_below_64_resets(self):
        config = Config(max_image_size=10)
        assert config.max_image_size == 4096

    def test_max_image_size_non_int_resets(self):
        config = Config(max_image_size="big")  # type: ignore[arg-type]
        assert config.max_image_size == 4096

    def test_valid_max_image_size_accepted(self):
        assert Config(max_image_size=64).max_image_size == 64
        assert Config(max_image_size=2048).max_image_size == 2048

    # ── type-mismatch inputs (list/dict/None from YAML) ──────────────────────

    def test_ocr_engine_as_list_resets(self):
        config = Config(ocr_engine=["tesseract"])  # type: ignore[arg-type]
        assert config.ocr_engine == "ensemble"

    def test_ocr_engine_as_none_resets(self):
        config = Config(ocr_engine=None)  # type: ignore[arg-type]
        assert config.ocr_engine == "ensemble"

    def test_display_server_as_dict_resets(self):
        config = Config(display_server={"value": "wayland"})  # type: ignore[arg-type]
        assert config.display_server == "auto"

    def test_confidence_threshold_as_string_resets(self):
        config = Config(ocr_confidence_threshold="high")  # type: ignore[arg-type]
        assert config.ocr_confidence_threshold == 0.6

    def test_confidence_threshold_as_none_resets(self):
        config = Config(ocr_confidence_threshold=None)  # type: ignore[arg-type]
        assert config.ocr_confidence_threshold == 0.6

    def test_history_size_as_bool_resets(self):
        config = Config(history_size=True)  # type: ignore[arg-type]
        assert config.history_size == 50

    def test_max_image_size_as_bool_resets(self):
        config = Config(max_image_size=True)  # type: ignore[arg-type]
        assert config.max_image_size == 4096

    def test_history_size_zero_resets(self):
        config = Config(history_size=0)
        assert config.history_size == 50

    def test_history_size_valid_accepted(self):
        assert Config(history_size=100).history_size == 100


class TestRenderConfig:
    def test_output_is_valid_yaml(self):
        import yaml

        output = Config()._render_config()
        data = yaml.safe_load(output)
        assert data["ocr_engine"] == "ensemble"

    def test_comments_present_for_key_fields(self):
        output = Config()._render_config()
        assert "# " in output
        assert "ocr_engine" in output
        assert "ensemble" in output

    def test_round_trip_preserves_values(self, tmp_path):
        c1 = Config(ocr_language="eng+rus", ocr_confidence_threshold=0.75)
        path = tmp_path / "config.yaml"
        c1.save(path)
        c2 = Config.load(path)
        assert c2.ocr_language == "eng+rus"
        assert c2.ocr_confidence_threshold == 0.75

    def test_all_fields_present_in_output(self):
        import dataclasses

        import yaml

        output = Config()._render_config()
        loaded_keys = set(yaml.safe_load(output).keys())
        expected_keys = {f.name for f in dataclasses.fields(Config)}
        assert expected_keys <= loaded_keys


class TestConfigProfiles:
    def test_list_profiles_empty_when_no_dir(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        assert Config.list_profiles(config_path) == []

    def test_list_profiles_returns_stems(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "fast.yaml").write_text("ocr_engine: tesseract\n")
        (profiles_dir / "gpu.yaml").write_text("use_gpu: true\n")
        assert Config.list_profiles(config_path) == ["fast", "gpu"]

    def test_load_with_profile_applies_override(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        Config().save(config_path)
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "fast.yaml").write_text("ocr_engine: tesseract\n")
        config = Config.load_with_profile(config_path, "fast")
        assert config.ocr_engine == "tesseract"

    def test_load_with_profile_keeps_base_fields(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        base = Config(ocr_language="rus")
        base.save(config_path)
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "fast.yaml").write_text("ocr_engine: tesseract\n")
        config = Config.load_with_profile(config_path, "fast")
        assert config.ocr_language == "rus"
        assert config.ocr_engine == "tesseract"

    def test_load_with_profile_missing_raises(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        with pytest.raises(FileNotFoundError, match="no-such"):
            Config.load_with_profile(config_path, "no-such")

    def test_load_with_profile_no_base_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "fast.yaml").write_text("ocr_engine: tesseract\n")
        config = Config.load_with_profile(config_path, "fast")
        assert config.ocr_engine == "tesseract"

    def test_missing_profile_does_not_create_base_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        with pytest.raises(FileNotFoundError):
            Config.load_with_profile(config_path, "no-such")
        assert not config_path.exists()

    def test_malformed_profile_yaml_raises(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "bad.yaml").write_text("- item1\n- item2\n")  # list, not mapping
        with pytest.raises(ValueError, match="mapping"):
            Config.load_with_profile(config_path, "bad")
