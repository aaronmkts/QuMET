"""Tests for JOSS and community scaffold metadata files."""
from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_project_scaffold_files_exist_with_required_shape():
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

    contributing = _read("CONTRIBUTING.md")
    for heading in [
        "# Contributing to QuMET",
        "## Development Setup",
        "## Running Tests",
        "## Reporting Issues",
        "## Pull Requests",
        "## Code Style",
    ]:
        assert heading in contributing

    citation = _read("CITATION.cff")
    for snippet in [
        "cff-version:",
        'title: "QuMET"',
        "license: Apache-2.0",
        "message:",
        "authors:",
        "repository-code:",
    ]:
        assert snippet in citation

    changelog = _read("CHANGELOG.md")
    assert "# Changelog" in changelog
    assert "## [Unreleased]" in changelog
    assert "JOSS" in changelog

    support = _read("SUPPORT.md")
    for snippet in [
        "# Support",
        "GitHub Issues",
        "feature",
        "discussion",
    ]:
        assert snippet in support

    paper = _read("paper/paper.md")
    for heading in [
        "# Summary",
        "# Statement of need",
        "# State of the field",
        "# Software design",
        "# Research impact statement",
        "# AI usage disclosure",
        "# Acknowledgements",
        "# References",
    ]:
        assert heading in paper

    bibliography = _read("paper/paper.bib")
    for citation_key in [
        "@article{bergholm2018pennylane",
        "@article{paszke2019pytorch",
        "@article{dallaire2018qgan",
        "@article{liu2018qcbm",
    ]:
        assert citation_key in bibliography

    pyproject = tomllib.loads(_read("pyproject.toml"))
    project = pyproject["project"]
    assert project["license"]["text"] == "Apache-2.0"
    assert "License :: OSI Approved :: Apache Software License" in project["classifiers"]
