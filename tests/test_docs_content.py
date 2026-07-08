"""Docs-content checks for JOSS-facing public documentation."""

import argparse
import re
import tomllib
from pathlib import Path

from qumet.cli import TASKS, QuMETCLI, _validate_model_dataset_task
from qumet.dataset import AVAILABLE_DATASETS, get_dataset_info
from qumet.models import get_model_info
from qumet.models.qcbm import QCBM_MODELS
from qumet.models.qgan import QGAN_MODELS
from qumet.models.vae import VAE_MODELS


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
PUBLIC_MARKDOWN_DOCS = [README_PATH, *sorted(DOCS_DIR.rglob("*.md"))]
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


def _get_cli_parser() -> argparse.ArgumentParser:
    return QuMETCLI.__new__(QuMETCLI)._setup_parser()


def _get_valid_cli_flags() -> set[str]:
    parser = _get_cli_parser()
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }


def _iter_raw_cli_flags(content: str) -> list[str]:
    return re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", content)


def _extract_markdown_section(content: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    match = re.search(pattern, content, flags=re.MULTILINE)
    assert match, f"Missing section heading: {heading}"

    following = content[match.end():]
    next_heading = re.search(r"^##\s+", following, flags=re.MULTILINE)
    end = next_heading.start() if next_heading else len(following)
    return following[:end]


def _parse_markdown_list_tokens(section: str) -> list[str]:
    tokens: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            tokens.extend(re.findall(r"`([^`]+)`", stripped))
    return list(dict.fromkeys(tokens))


def _documented_model_names(content: str) -> list[str]:
    section = _extract_markdown_section(content, "Supported Models")
    return _parse_markdown_list_tokens(section)


def _documented_dataset_names(content: str) -> list[str]:
    section = _extract_markdown_section(content, "Supported Datasets")
    return _parse_markdown_list_tokens(section)


def _documented_task_names(content: str) -> list[str]:
    section = _extract_markdown_section(content, "Supported Tasks")
    return _parse_markdown_list_tokens(section)


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


def test_public_docs_use_valid_cli_option_names():
    valid_flags = _get_valid_cli_flags()
    invalid_flags = {}

    for path in PUBLIC_MARKDOWN_DOCS:
        content = _read(path)
        bad_flags = sorted(
            {
                flag
                for flag in _iter_raw_cli_flags(content)
                if "_" in flag and flag not in valid_flags and flag.replace("_", "-") in valid_flags
            }
        )
        if bad_flags:
            invalid_flags[str(path.relative_to(REPO_ROOT))] = bad_flags

    assert not invalid_flags, (
        "Public Markdown docs use invalid CLI option spellings; use the CLI-defined "
        f"hyphenated flags instead: {invalid_flags}"
    )


def test_supported_name_lists_match_current_registries():
    expected_models = sorted(
        [*QCBM_MODELS.keys(), *QGAN_MODELS.keys(), *VAE_MODELS.keys()]
    )
    expected_datasets = sorted(AVAILABLE_DATASETS)
    expected_tasks = sorted(TASKS)

    docs_to_check = [
        README_PATH,
        DOCS_DIR / "models-and-datasets.md",
    ]

    mismatches = {}
    for path in docs_to_check:
        content = _read(path)
        actual = {
            "models": sorted(_documented_model_names(content)),
            "datasets": sorted(_documented_dataset_names(content)),
            "tasks": sorted(_documented_task_names(content)),
        }
        expected = {
            "models": expected_models,
            "datasets": expected_datasets,
            "tasks": expected_tasks,
        }
        for label, expected_values in expected.items():
            if actual[label] != expected_values:
                mismatches.setdefault(str(path.relative_to(REPO_ROOT)), {})[label] = {
                    "expected": expected_values,
                    "actual": actual[label],
                }

    assert not mismatches, f"Documented supported names drifted from code: {mismatches}"
