"""Tests for qumet.tools.config_load module."""
import pytest
import tempfile
import os
from argparse import Namespace
from qumet.tools.config_load import (
    convert_str_na_to_none,
    convert_none_to_str_na,
    load_config,
    save_config,
    post_parse_load_config,
)


class TestConvertStrNaToNone:
    """Tests for convert_str_na_to_none."""

    def test_converts_na_string_to_none(self):
        result = convert_str_na_to_none({"key": "NA"})
        assert result["key"] is None

    def test_leaves_other_strings_unchanged(self):
        result = convert_str_na_to_none({"key": "hello"})
        assert result["key"] == "hello"

    def test_handles_nested_dicts(self):
        result = convert_str_na_to_none({"outer": {"inner": "NA"}})
        assert result["outer"]["inner"] is None

    def test_handles_lists(self):
        result = convert_str_na_to_none({"items": ["NA", "keep", "NA"]})
        assert result["items"] == [None, "keep", None]

    def test_handles_tuples(self):
        result = convert_str_na_to_none({"items": ("NA", "keep")})
        assert result["items"] == (None, "keep")

    def test_handles_non_dict_input(self):
        assert convert_str_na_to_none("NA") is None
        assert convert_str_na_to_none("hello") == "hello"
        assert convert_str_na_to_none(42) == 42


class TestConvertNoneToStrNa:
    """Tests for convert_none_to_str_na."""

    def test_converts_none_to_na_string(self):
        result = convert_none_to_str_na({"key": None})
        assert result["key"] == "NA"

    def test_leaves_other_values_unchanged(self):
        result = convert_none_to_str_na({"key": "hello"})
        assert result["key"] == "hello"

    def test_handles_nested_dicts(self):
        result = convert_none_to_str_na({"outer": {"inner": None}})
        assert result["outer"]["inner"] == "NA"

    def test_handles_lists(self):
        result = convert_none_to_str_na({"items": [None, "keep", None]})
        assert result["items"] == ["NA", "keep", "NA"]

    def test_handles_tuples(self):
        result = convert_none_to_str_na({"items": (None, "keep")})
        assert result["items"] == ("NA", "keep")

    def test_roundtrip_identity(self):
        """convert_str_na_to_none . convert_none_to_str_na should be identity for NA/None."""
        original = {"a": "NA", "b": {"c": "NA"}, "d": ["NA", "x"]}
        roundtripped = convert_none_to_str_na(convert_str_na_to_none(original))
        assert roundtripped == original


class TestLoadSaveConfig:
    """Tests for load_config and save_config."""

    def test_load_config_converts_na(self, tmp_path):
        config_path = tmp_path / "test.toml"
        config_path.write_text("[section]\nkey = \"NA\"\nother = \"value\"\n")
        config = load_config(str(config_path))
        assert config["section"]["key"] is None
        assert config["section"]["other"] == "value"

    def test_save_config_converts_none(self, tmp_path):
        config_path = tmp_path / "test.toml"
        config = {"section": {"key": None, "other": "value"}}
        save_config(config, str(config_path))
        loaded = load_config(str(config_path))
        assert loaded["section"]["key"] is None
        assert loaded["section"]["other"] == "value"

    def test_load_config_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.toml")


class TestPostParseLoadConfig:
    """Tests for post_parse_load_config."""

    DEFAULTS = {
        "model": None,
        "dataset": None,
        "batch_size": 8,
        "seed": 42,
        "learning_rate": 1e-5,
        "config": None,
    }

    def test_no_config_returns_args_unchanged(self):
        args = Namespace(**self.DEFAULTS)
        result = post_parse_load_config(args, self.DEFAULTS)
        assert result.batch_size == 8
        assert result.seed == 42

    def test_config_overrides_defaults(self, tmp_path):
        config_path = tmp_path / "cfg.toml"
        config_path.write_text("batch_size = 16\nseed = 99\n")
        args = Namespace(**{**self.DEFAULTS, "config": str(config_path)})
        result = post_parse_load_config(args, self.DEFAULTS)
        assert result.batch_size == 16
        assert result.seed == 99

    def test_manual_override_wins_over_config(self, tmp_path):
        config_path = tmp_path / "cfg.toml"
        config_path.write_text("batch_size = 16\n")
        args = Namespace(**{**self.DEFAULTS, "config": str(config_path), "batch_size": 32})
        result = post_parse_load_config(args, self.DEFAULTS)
        assert result.batch_size == 32  # manual override wins

    def test_rejects_non_toml_config(self, tmp_path):
        config_path = tmp_path / "cfg.json"
        config_path.write_text("{}")
        args = Namespace(**{**self.DEFAULTS, "config": str(config_path)})
        with pytest.raises(ValueError, match="expected .toml"):
            post_parse_load_config(args, self.DEFAULTS)

    def test_metrics_section_loaded(self, tmp_path):
        config_path = tmp_path / "cfg.toml"
        config_path.write_text("[metrics]\nuse_metrics = [\"m1\", \"m2\"]\n")
        args = Namespace(**{**self.DEFAULTS, "config": str(config_path)})
        result = post_parse_load_config(args, self.DEFAULTS)
        assert result.metrics_to_use == ["m1", "m2"]

    def test_ignores_keys_not_in_defaults(self):
        args = Namespace(**{**self.DEFAULTS, "extra_field": "ignored"})
        result = post_parse_load_config(args, self.DEFAULTS)
        # extra_field should be untouched
        assert result.extra_field == "ignored"
