"""Docs-content checks for JOSS-facing public documentation."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
DOCS_DIR = REPO_ROOT / "docs"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_covers_joss_public_surface():
    readme = _read(README_PATH)

    required_phrases = [
        "unified torch/pennylane framework",
        "comparing qgans and qcbms from the literature",
        "statement of need",
        "target audience",
        "pip install -e .",
        "quickstart",
        "supported models",
        "supported datasets",
        "supported tasks",
        "pytest",
        "configuration",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in readme.lower()]
    assert not missing, f"README is missing required public-facing content: {missing}"
    assert "search functionality is still in progress" not in readme.lower()


def test_joss_readme_links_to_detail_docs():
    readme = _read(README_PATH)

    expected_docs = [
        "docs/quickstart.md",
        "docs/configuration.md",
        "docs/models-and-datasets.md",
        "docs/testing.md",
    ]

    missing = [doc for doc in expected_docs if doc not in readme]
    assert not missing, f"README is missing links to detailed docs: {missing}"


def test_detailed_docs_exist_and_cover_current_examples():
    doc_expectations = {
        DOCS_DIR / "quickstart.md": ["qcbm", "2d_gaussian", "qmt train"],
        DOCS_DIR / "configuration.md": ["toml", "src/configs/by_model", "[metrics]"],
        DOCS_DIR / "models-and-datasets.md": ["pqwgan_qc", "mnist", "discrete_generation"],
        DOCS_DIR / "testing.md": ["pytest", "tests/test_docs_content.py"],
    }

    missing_files = [str(path.relative_to(REPO_ROOT)) for path in doc_expectations if not path.exists()]
    assert not missing_files, f"Expected detailed docs are missing: {missing_files}"

    missing_content = {}
    for path, phrases in doc_expectations.items():
        content = _read(path).lower()
        absent = [phrase for phrase in phrases if phrase.lower() not in content]
        if absent:
            missing_content[str(path.relative_to(REPO_ROOT))] = absent

    assert not missing_content, f"Detailed docs are missing required content: {missing_content}"
