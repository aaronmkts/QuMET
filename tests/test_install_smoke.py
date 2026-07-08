"""Smoke tests for source-checkout importability and package metadata."""

from pathlib import Path
import tomllib

import qumet


def test_source_checkout_imports_qumet_package():
    package_root = Path(qumet.__file__).resolve().parents[1]

    assert package_root.name == "src"


def test_package_version_matches_project_metadata():
    pyproject_data = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject_data["project"]["name"] == "qumet"
    assert qumet.__version__ == pyproject_data["project"]["version"]
