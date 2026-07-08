"""Smoke tests for importability and package metadata."""

from pathlib import Path
import tomllib

import qumet
from qumet import models


def test_source_checkout_imports_qumet_package():
    assert isinstance(qumet.__version__, str)
    assert qumet.__version__

    qcbm_info = models.get_model_info("qcbm")

    assert qcbm_info.name == "qcbm"
    assert qcbm_info.model_type.value == "qcbm"
    assert qcbm_info.task_type.value == "generation"
    assert qcbm_info.bitstring_sampling is True
    assert qcbm_info.observable_sampling is False
    assert qcbm_info.is_generation_model is True


def test_package_version_matches_project_metadata():
    pyproject_data = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject_data["project"]["version"]
    assert qumet.__version__ == pyproject_data["project"]["version"]
