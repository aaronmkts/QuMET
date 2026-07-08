# JOSS Workstream 5 Report

## Internal Plan

1. Inspect the current package metadata and repository structure that the JOSS/community scaffold must match.
2. Add a focused scaffold test that fails until the required metadata and paper files exist with the expected headings and key fields.
3. Implement the minimal documentation and metadata files in scope:
   - `CONTRIBUTING.md`
   - `CITATION.cff`
   - `CHANGELOG.md`
   - `SUPPORT.md`
   - `paper/paper.md`
   - `paper/paper.bib`
4. Align package metadata that directly conflicts with the scaffold requirements, especially the Apache-2.0 license reference in `pyproject.toml`.
5. Run the focused test first, then the broader test suite if feasible with `/home/aaron/miniconda3/envs/qumet/bin/python`.
6. Commit the resulting changes on `joss/scaffold` without touching unrelated work.

## TDD Evidence

### RED

Command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_project_scaffold_metadata.py -v
```

Relevant output:

```text
tests/test_project_scaffold_metadata.py::test_project_scaffold_files_exist_with_required_shape FAILED
E   AssertionError: Missing required scaffold file: CONTRIBUTING.md
```

### GREEN

Focused command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_project_scaffold_metadata.py -v
```

Relevant output:

```text
tests/test_project_scaffold_metadata.py::test_project_scaffold_files_exist_with_required_shape PASSED
============================== 1 passed in 0.03s ===============================
```

Repo-wide verification:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest
```

Relevant output:

```text
======================= 155 passed, 2 warnings in 9.91s ========================
```

Additional note:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest
```

This initial full-suite attempt failed during collection with `ModuleNotFoundError: No module named 'qumet'`, so the successful repo-wide verification used `PYTHONPATH=src`, which matches the package layout under `src/qumet`.

## Files Changed

- `CONTRIBUTING.md`
- `CITATION.cff`
- `CHANGELOG.md`
- `SUPPORT.md`
- `paper/paper.md`
- `paper/paper.bib`
- `tests/test_project_scaffold_metadata.py`
- `pyproject.toml`
- `JOSS_WORKSTREAM_REPORT.md`

## Commits Created

- `docs: add JOSS submission scaffold`

## Verification Commands

- `/home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_project_scaffold_metadata.py -v`
- `/home/aaron/miniconda3/envs/qumet/bin/python -m pytest`
- `PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest`

## Concerns

- The repository's default `pytest` invocation does not currently place `src/` on `PYTHONPATH`, so repo-wide verification required `PYTHONPATH=src`.
- `CITATION.cff` and `paper/paper.bib` are conservative starting points and may need maintainer refinement for author list, preferred citation, and manuscript wording before submission.
