"""Shared fixtures for QuMET test suite."""
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory that cleans up after test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_toml_config(temp_dir):
    """Create a temporary TOML config file."""
    config_path = temp_dir / "config.toml"
    config_path.write_text("""
[training]
batch_size = 16
learning_rate = 0.001
seed = 99
""")
    return str(config_path)
