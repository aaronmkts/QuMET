"""Tests for qumet.tools.registry module."""
import pytest
from pathlib import Path
from qumet.tools.registry import MAIN_DIR, MAIN_CACHE_DIR


class TestRegistryPaths:
    """Tests for registry path constants."""

    def test_main_dir_is_absolute(self):
        assert MAIN_DIR.is_absolute()

    def test_main_dir_exists(self):
        assert MAIN_DIR.exists()

    def test_main_cache_dir_is_under_main(self):
        assert str(MAIN_CACHE_DIR).startswith(str(MAIN_DIR))

    def test_main_cache_dir_name(self):
        assert MAIN_CACHE_DIR.name == ".qumet_cache"
