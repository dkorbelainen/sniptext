"""Tests for Config."""

import pytest

from sniptext.config import Config


class TestConfigDefaults:
    def test_max_image_size_default(self):
        assert Config().max_image_size == 4096

    def test_ocr_language(self):
        assert Config().ocr_language == "eng"


class TestConfigSaveLoad:
    def test_round_trip(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        original = Config()
        original.save(config_path)

        loaded = Config.load(config_path)

        assert loaded.ocr_language == original.ocr_language
        assert loaded.routing == original.routing

    def test_load_creates_file_if_missing(self, tmp_path):
        config_path = tmp_path / "new_config.yaml"
        assert not config_path.exists()

        config = Config.load(config_path)

        assert config_path.exists()
        assert config.ocr_language == "eng"

    def test_save_load_custom_values(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        original = Config(ocr_language="rus", max_image_size=1024)
        original.save(config_path)

        loaded = Config.load(config_path)

        assert loaded.ocr_language == "rus"
        assert loaded.max_image_size == 1024

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
        assert config.ocr_language == "eng"
        assert not hasattr(config, "totally_unknown_key")


class TestConfigValidation:
    def test_invalid_display_server_resets_to_auto(self):
        config = Config(display_server="foobar")
        assert config.display_server == "auto"

    def test_valid_display_server_accepted(self):
        for ds in ("auto", "wayland", "x11"):
            assert Config(display_server=ds).display_server == ds

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

    def test_display_server_as_dict_resets(self):
        config = Config(display_server={"value": "wayland"})  # type: ignore[arg-type]
        assert config.display_server == "auto"

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
        assert data["ocr_language"] == "eng"

    def test_comments_present_for_key_fields(self):
        output = Config()._render_config()
        assert "# " in output
        assert "ocr_language" in output
        assert "routing" in output

    def test_round_trip_preserves_values(self, tmp_path):
        c1 = Config(ocr_language="eng+rus", max_image_size=2048)
        path = tmp_path / "config.yaml"
        c1.save(path)
        c2 = Config.load(path)
        assert c2.ocr_language == "eng+rus"
        assert c2.max_image_size == 2048

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
        (profiles_dir / "fast.yaml").write_text("ocr_language: rus\n")
        (profiles_dir / "gpu.yaml").write_text("routing: false\n")
        assert Config.list_profiles(config_path) == ["fast", "gpu"]

    def test_load_with_profile_applies_override(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        Config().save(config_path)
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "fast.yaml").write_text("ocr_language: rus\n")
        config = Config.load_with_profile(config_path, "fast")
        assert config.ocr_language == "rus"

    def test_load_with_profile_keeps_base_fields(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        base = Config(ocr_language="rus")
        base.save(config_path)
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "fast.yaml").write_text("routing: false\n")
        config = Config.load_with_profile(config_path, "fast")
        assert config.ocr_language == "rus"
        assert config.routing is False

    def test_load_with_profile_missing_raises(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        with pytest.raises(FileNotFoundError, match="no-such"):
            Config.load_with_profile(config_path, "no-such")

    def test_load_with_profile_no_base_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "fast.yaml").write_text("ocr_language: rus\n")
        config = Config.load_with_profile(config_path, "fast")
        assert config.ocr_language == "rus"

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


class TestRouterConfig:
    def test_time_weight_defaults_to_the_benchmarked_value(self):
        assert Config().router_time_weight is None

    def test_time_weight_accepts_non_negative_numbers(self):
        assert Config(router_time_weight=0).router_time_weight == 0.0
        assert Config(router_time_weight="0.25").router_time_weight == 0.25

    def test_invalid_time_weight_resets_to_none(self):
        assert Config(router_time_weight=-1).router_time_weight is None
        assert Config(router_time_weight="fast").router_time_weight is None
        assert Config(router_time_weight=True).router_time_weight is None

    def test_time_weight_round_trip(self, tmp_path):
        path = tmp_path / "config.yaml"
        Config(router_time_weight=0.5).save(path)
        assert Config.load(path).router_time_weight == 0.5
        Config().save(path)
        assert Config.load(path).router_time_weight is None

    def test_removed_ab_test_key_is_dropped(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("ocr_language: rus\nab_test_probability: 0.15\n")
        config = Config.load(path)
        assert config.ocr_language == "rus"
        assert not hasattr(config, "ab_test_probability")


class TestRoutingConfig:
    def test_routing_is_on_by_default(self):
        assert Config().routing is True

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_non_boolean_routing_is_reset(self, value):
        assert Config(routing=value).routing is True

    def test_routing_can_be_switched_off(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("routing: false\n")
        assert Config.load(path).routing is False

    def test_a_0_4_config_file_still_loads(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "hotkey: <ctrl>+<alt>+t\nocr_engine: ensemble\nocr_language: eng+rus+ell+equ\n"
            "ocr_confidence_threshold: 0.6\nadaptive_ensemble: true\nmax_image_size: 4096\n"
            "use_gpu: true\nnotification_enabled: true\nenable_text_correction: true\n"
            "aggressive_correction: false\n"
        )
        config = Config.load(path)
        assert config.ocr_language == "eng+rus+ell+equ" and config.routing is True
        assert config.max_image_size == 4096 and config.notification_enabled is True

    def test_rendered_config_has_no_removed_keys(self):
        rendered = Config()._render_config()
        for key in ("ocr_engine", "ocr_model_path", "use_gpu", "adaptive_ensemble",
                    "ocr_confidence_threshold", "enable_text_correction",
                    "aggressive_correction", "hotkey"):  # fmt: skip
            assert key not in rendered
        assert "routing: true" in rendered
