"""Tests for JOSS and community scaffold metadata files."""

from pathlib import Path
import re
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _markdown_headings(text: str) -> list[tuple[int, str]]:
    return [
        (len(match.group(1)), match.group(2).strip())
        for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, flags=re.MULTILINE)
    ]


def _bib_entries(text: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("entry_type").lower()
        for match in re.finditer(
            r"@(?P<entry_type>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,",
            text,
        )
    }


def test_project_scaffold_files_exist():
    required_files = [
        "CONTRIBUTING.md",
        "CITATION.cff",
        "CHANGELOG.md",
        "SUPPORT.md",
        "paper/paper.md",
        "paper/paper.bib",
    ]

    for relative_path in required_files:
        assert (REPO_ROOT / relative_path).exists(), f"Missing required scaffold file: {relative_path}"


def test_contributing_has_required_sections_and_branch_accurate_test_commands():
    contributing = _read("CONTRIBUTING.md")
    headings = _markdown_headings(contributing)

    assert headings[:1] == [(1, "Contributing to QuMET")]
    assert [heading for level, heading in headings if level == 2] == [
        "Development Setup",
        "Running Tests",
        "Reporting Issues",
        "Pull Requests",
        "Code Style",
    ]
    assert "python -m pip install -e '.[dev]'" in contributing
    assert "python -m pytest" in contributing
    assert "PYTHONPATH=src python -m pytest" in contributing
    assert "python -m pytest tests/test_project_scaffold_metadata.py -v" in contributing
    assert "/home/aaron/" not in contributing


def test_citation_cff_parses_and_contains_required_metadata():
    citation = yaml.safe_load(_read("CITATION.cff"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["title"] == "QuMET"
    assert citation["license"] == "Apache-2.0"
    assert citation["repository-code"] == "https://github.com/aaronmkts/QuMET"
    assert citation["url"] == "https://github.com/aaronmkts/QuMET"
    assert citation["message"]
    assert citation["abstract"]

    authors = citation["authors"]
    assert isinstance(authors, list) and authors
    assert authors[0]["family-names"] == "Thomas"
    assert authors[0]["given-names"] == "Aaron M."

    keywords = citation["keywords"]
    assert isinstance(keywords, list)
    assert {"PennyLane", "PyTorch", "QGAN"}.issubset(keywords)


def test_support_and_changelog_have_required_sections():
    support_headings = _markdown_headings(_read("SUPPORT.md"))
    assert support_headings[:1] == [(1, "Support")]
    assert [heading for level, heading in support_headings if level == 2] == [
        "Ask for Help",
        "Report Bugs",
        "Request Features",
    ]

    support = _read("SUPPORT.md")
    assert "GitHub Issues" in support
    assert "question:" in support
    assert "discussion:" in support

    changelog_headings = _markdown_headings(_read("CHANGELOG.md"))
    assert changelog_headings[:1] == [(1, "Changelog")]
    assert (2, "[Unreleased]") in changelog_headings
    assert (3, "Added") in changelog_headings
    assert (3, "Changed") in changelog_headings
    assert "JOSS/community submission scaffold" in _read("CHANGELOG.md")


def test_paper_headings_and_bibliography_entries_match_expected_shape():
    paper_headings = _markdown_headings(_read("paper/paper.md"))
    assert [heading for level, heading in paper_headings if level == 1] == [
        "Summary",
        "Statement of need",
        "State of the field",
        "Software design",
        "Research impact statement",
        "AI usage disclosure",
        "Acknowledgements",
        "References",
    ]

    bibliography = _bib_entries(_read("paper/paper.bib"))
    assert bibliography == {
        "bergholm2018pennylane": "article",
        "paszke2019pytorch": "article",
        "dallaire2018qgan": "article",
        "liu2018qcbm": "article",
    }


def test_pyproject_license_metadata_matches_repository_license():
    pyproject = tomllib.loads(_read("pyproject.toml"))
    project = pyproject["project"]
    assert project["license"]["text"] == "Apache-2.0"
    assert "License :: OSI Approved :: Apache Software License" in project["classifiers"]
