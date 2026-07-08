"""Production-readiness checks for QGAN-family models."""

import ast
from pathlib import Path

import pytest

from qumet.dataset import get_dataset_info
from qumet.models import get_model
from qumet.models.qgan import QGAN_MODELS, get_qgan_model_info


REPO_ROOT = Path(__file__).resolve().parents[1]
QGAN_DOCS_PATH = REPO_ROOT / "docs" / "qgan-models.md"

PRODUCTION_QGAN_MODELS = ("patchgan", "mosaiq", "pqwgan_qc", "qinr", "gan")
DEFERRED_QGAN_MODELS = {"vaeqwgan"}

QGAN_MODULE_PATHS = {
    "patchgan": REPO_ROOT / "src/qumet/models/qgan/patchgan/modelling_patchgan.py",
    "mosaiq": REPO_ROOT / "src/qumet/models/qgan/mosaiq/modelling_mosaiq.py",
    "pqwgan_qc": REPO_ROOT / "src/qumet/models/qgan/pwqgan/modelling_pwqgan_qc.py",
    "qinr": REPO_ROOT / "src/qumet/models/qgan/qinr/modelling_qinr.py",
    "gan": REPO_ROOT / "src/qumet/models/qgan/classical_gan/gan.py",
}


@pytest.mark.parametrize("model_name", PRODUCTION_QGAN_MODELS)
def test_production_qgan_models_construct_for_image_generation(model_name):
    """Every production-facing QGAN/GAN registry entry should instantiate."""
    model = get_model(model_name, "image_generation", get_dataset_info("mnist"))

    assert model is not None


def test_qgan_registry_deferred_models_are_explicit():
    """Deferred QGAN models should stay visible while production models are tested."""
    assert set(QGAN_MODELS) == {*PRODUCTION_QGAN_MODELS, *DEFERRED_QGAN_MODELS}
    for model_name in PRODUCTION_QGAN_MODELS:
        assert get_qgan_model_info(model_name).name == model_name


def test_mosaiq_constructor_matches_reference_structure():
    """MosaiQ should preserve the paper/source-level generator structure."""
    model = get_model("mosaiq", "image_generation", get_dataset_info("mnist"))

    assert model.discriminator.input_size == 40
    assert model.generator.n_generators == 8
    assert model.generator.n_qubits == 5
    assert model.generator.depth == 6
    assert model.generator.feature_redistribution() == [
        [0, 39, 38, 37, 36],
        [1, 35, 34, 33, 32],
        [2, 31, 30, 29, 28],
        [3, 27, 26, 25, 24],
        [4, 23, 22, 21, 20],
        [5, 19, 18, 17, 16],
        [6, 15, 14, 13, 12],
        [7, 11, 10, 9, 8],
    ]


def test_qgan_model_documentation_covers_production_models():
    """Production-facing QGAN/GAN models need a reviewer-facing docs page."""
    content = QGAN_DOCS_PATH.read_text(encoding="utf-8")

    required_sections = [
        "# QGAN and GAN Models",
        "## Production-Facing Models",
        "## Deferred Models",
        "## Testing Expectations",
    ]
    for section in required_sections:
        assert section in content

    for model_name in PRODUCTION_QGAN_MODELS:
        assert f"`{model_name}`" in content


def test_qgan_model_modules_have_google_style_docstrings():
    """Public QGAN modules/classes/functions should carry Google-style docstrings."""
    missing = {}

    for model_name, path in QGAN_MODULE_PATHS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_missing = []
        if not ast.get_docstring(tree):
            module_missing.append("<module>")

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not ast.get_docstring(node):
                module_missing.append(node.name)
            if (
                isinstance(node, ast.FunctionDef)
                and not node.name.startswith("_")
                and not ast.get_docstring(node)
            ):
                module_missing.append(node.name)

        if module_missing:
            missing[model_name] = module_missing

    assert not missing, f"Missing QGAN docstrings: {missing}"
