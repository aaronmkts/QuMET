"""Tests for qumet.cli module."""
import pytest
import argparse
import tempfile
import os
from qumet.cli import (
    QuMETCLI,
    _valid_filepath,
    _valid_directory_path,
    _valid_file_or_directory_path,
    _positive_int,
    _int,
    ShowVersionAction,
    ShowInfoAction,
    CLI_DEFAULTS,
    VERSION,
)


class TestValidFilepath:
    """Tests for _valid_filepath."""

    def test_returns_absolute_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _valid_filepath(str(f))
        assert os.path.isabs(result)
        assert result == str(f.resolve())

    def test_raises_on_missing_file(self):
        with pytest.raises(argparse.ArgumentTypeError, match="file not found"):
            _valid_filepath("/nonexistent/file.txt")

    def test_raises_on_directory(self, tmp_path):
        with pytest.raises(argparse.ArgumentTypeError, match="expected path to file"):
            _valid_filepath(str(tmp_path))


class TestValidDirectoryPath:
    """Tests for _valid_directory_path."""

    def test_returns_absolute_path(self, tmp_path):
        result = _valid_directory_path(str(tmp_path))
        assert os.path.isabs(result)

    def test_raises_on_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(argparse.ArgumentTypeError, match="expected path to directory"):
            _valid_directory_path(str(f))

    def test_raises_on_missing_without_create(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(argparse.ArgumentTypeError, match="directory not found"):
            _valid_directory_path(str(missing), create_dir=False)

    def test_creates_missing_with_create(self, tmp_path):
        missing = tmp_path / "new_dir"
        result = _valid_directory_path(str(missing), create_dir=True)
        assert os.path.isdir(missing)
        assert os.path.isabs(result)


class TestValidFileOrDirectoryPath:
    """Tests for _valid_file_or_directory_path."""

    def test_accepts_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _valid_file_or_directory_path(str(f))
        assert os.path.isabs(result)

    def test_accepts_directory(self, tmp_path):
        result = _valid_file_or_directory_path(str(tmp_path))
        assert os.path.isabs(result)

    def test_raises_on_missing(self):
        with pytest.raises(argparse.ArgumentTypeError, match="file or directory not found"):
            _valid_file_or_directory_path("/nonexistent/path")


class TestPositiveInt:
    """Tests for _positive_int."""

    def test_returns_int_for_positive(self):
        assert _positive_int("5") == 5
        assert _positive_int("100") == 100

    def test_returns_none_for_zero(self):
        assert _positive_int("0") is None

    def test_returns_none_for_negative(self):
        assert _positive_int("-5") is None

    def test_raises_on_non_integer(self):
        with pytest.raises(argparse.ArgumentError):
            _positive_int("abc")


class TestInt:
    """Tests for _int."""

    def test_returns_int(self):
        assert _int("5") == 5
        assert _int("-3") == -3
        assert _int("0") == 0

    def test_raises_on_non_integer(self):
        with pytest.raises(argparse.ArgumentError):
            _int("abc")


class TestCLIDefaults:
    """Tests for CLI_DEFAULTS constant."""

    def test_has_required_keys(self):
        required = ["model", "dataset", "batch_size", "seed", "learning_rate",
                     "max_epochs", "accelerator", "strategy"]
        for key in required:
            assert key in CLI_DEFAULTS, f"Missing key: {key}"

    def test_default_seed_is_42(self):
        assert CLI_DEFAULTS["seed"] == 42

    def test_default_batch_size_is_8(self):
        assert CLI_DEFAULTS["batch_size"] == 8


class TestShowVersionAction:
    """Tests for ShowVersionAction."""

    def test_prints_and_exits(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-V", "--version", action=ShowVersionAction, help="show version and exit")
        with pytest.raises(SystemExit):
            parser.parse_args(["-V"])


class TestShowInfoAction:
    """Tests for ShowInfoAction."""

    def test_prints_and_exits(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--info", action=ShowInfoAction, const="all", choices=["all", "model", "dataset"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--info"])


class TestQuMETCLI:
    """Tests for QuMETCLI class."""

    def test_requires_model_and_dataset(self):
        """CLI should raise when model/dataset not provided."""
        with pytest.raises(ValueError, match="No model and/or dataset"):
            QuMETCLI(["train"])

    def test_version_constant(self):
        assert VERSION == "00.00.0"

    def test_parser_has_help(self):
        with pytest.raises(SystemExit):
            cli = QuMETCLI(["train", "--help"])
        # --help causes SystemExit
        with pytest.raises(SystemExit):
            QuMETCLI(["train", "--help"])
