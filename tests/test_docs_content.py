"""Docs-content checks for JOSS-facing public documentation."""

import re
import tomllib
from pathlib import Path

from qumet.cli import _validate_model_dataset_task
from qumet.dataset import get_dataset_info
from qumet.models import get_model_info


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
DOCS_DIR = REPO_ROOT / "docs"
PUBLIC_DOCS = [
    README_PATH,
    DOCS_DIR / "quickstart.md",
    DOCS_DIR / "configuration.md",
    DOCS_DIR / "models-and-datasets.md",
    DOCS_DIR / "testing.md",
]
README_REQUIRED_SECTIONS = [
    "# QuMET",
    "## Statement of Need",
    "## Target Audience",
    "## Installation",
    "## Quickstart",
    "## Supported Models",
    "## Supported Datasets",
    "## Supported Tasks",
    "## Configuration",
    "## Testing",
    "## Detailed Docs",
]
README_REQUIRED_LINKS = [
    "docs/quickstart.md",
    "docs/configuration.md",
    "docs/models-and-datasets.md",
    "docs/testing.md",
]
DOC_SECTION_REQUIREMENTS = {
    DOCS_DIR / "quickstart.md": [
        "# QuMET Quickstart",
        "## 1. Install QuMET",
        "## 2. Run a current example config",
        "## 3. Override values from the CLI",
        "## 4. Explore another current model family",
        "## 5. Read the next docs",
    ],
    DOCS_DIR / "configuration.md": [
        "# QuMET Configuration",
        "## Merge Order",
        "## Core Top-Level Keys",
        "## Metrics Subsections",
        "## Where Example Configs Live",
        "## Recommended Workflow",
    ],
    DOCS_DIR / "models-and-datasets.md": [
        "# QuMET Models and Datasets",
        "## Supported Models",
        "## Supported Datasets",
        "## Supported Tasks",
        "## Practical Pairings",
    ],
    DOCS_DIR / "testing.md": [
        "# QuMET Testing",
        "## Install test dependencies",
        "## Full Test Suite",
        "## Focused Docs Check",
        "## Useful Focused Commands",
    ],
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _assert_compatible(model: str, dataset: str, task: str) -> None:
    _validate_model_dataset_task(
        get_model_info(model),
        get_dataset_info(dataset),
        model,
        dataset,
        task,
    )


def _iter_referenced_configs(content: str) -> list[Path]:
    matches = re.findall(r"src/configs/[A-Za-z0-9_./-]+\.toml", content)
    return [REPO_ROOT / match for match in matches]


def _iter_documented_cli_triples(content: str) -> list[tuple[str, str, str]]:
    return re.findall(
        r"python\s+src/qmt\s+train\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s+--task\s+([A-Za-z0-9_]+)",
        content,
    )


def test_readme_covers_joss_public_surface():
    readme = _read(README_PATH)

    missing_sections = [section for section in README_REQUIRED_SECTIONS if section not in readme]
    assert not missing_sections, f"README is missing required sections: {missing_sections}"

    missing_links = [link for link in README_REQUIRED_LINKS if link not in readme]
    assert not missing_links, f"README is missing links to detailed docs: {missing_links}"

    assert "pip install -e '.[dev]'" in readme
    assert "python -m pytest" in readme
    assert "search functionality is still in progress" not in readme.lower()


def test_detailed_docs_have_expected_structure():
    missing_files = [str(path.relative_to(REPO_ROOT)) for path in DOC_SECTION_REQUIREMENTS if not path.exists()]
    assert not missing_files, f"Expected detailed docs are missing: {missing_files}"

    missing_sections = {}
    for path, sections in DOC_SECTION_REQUIREMENTS.items():
        content = _read(path)
        absent = [section for section in sections if section not in content]
        if absent:
            missing_sections[str(path.relative_to(REPO_ROOT))] = absent

    assert not missing_sections, f"Detailed docs are missing required sections: {missing_sections}"


def test_public_docs_reference_existing_compatible_examples():
    referenced_configs = {}
    documented_cli_triples = {}

    for path in PUBLIC_DOCS:
        content = _read(path)
        referenced_configs[path] = _iter_referenced_configs(content)
        documented_cli_triples[path] = _iter_documented_cli_triples(content)

    assert any(referenced_configs.values()), "Public docs should reference at least one example config"
    assert any(documented_cli_triples.values()), "Public docs should document at least one CLI training example"

    for path, config_paths in referenced_configs.items():
        for config_path in config_paths:
            assert config_path.exists(), f"Referenced config does not exist: {config_path.relative_to(REPO_ROOT)}"
            config = _load_toml(config_path)
            if {"model", "dataset", "task"} <= config.keys():
                _assert_compatible(config["model"], config["dataset"], config["task"])

    for path, triples in documented_cli_triples.items():
        for model, dataset, task in triples:
            _assert_compatible(model, dataset, task)
