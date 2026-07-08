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
5. Run the focused test first, then the broader test suite with the branch-accurate `PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest` invocation.
6. Commit the resulting changes on `joss/scaffold` without touching unrelated work.

## TDD Evidence

### RED

Command:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_project_scaffold_metadata.py -v
```

Relevant output:

```text
tests/test_project_scaffold_metadata.py::test_contributing_has_required_sections_and_branch_accurate_test_commands FAILED
E   assert 'PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest' in contributing
```

### GREEN

Focused command:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_project_scaffold_metadata.py -v
```

Relevant output:

```text
============================== 6 passed in 0.04s ===============================
```

Repo-wide verification:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest
```

Relevant output:

```text
======================= 160 passed, 2 warnings in 9.66s ========================
```

## Files Changed

- `CONTRIBUTING.md`
- `tests/test_project_scaffold_metadata.py`
- `JOSS_WORKSTREAM_REPORT.md`

## Commits Created

- `docs: fix JOSS scaffold review findings`

## Verification Commands

- `PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_project_scaffold_metadata.py -v`
- `PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest`

## Concerns

- The repository's default `pytest` invocation does not currently place `src/` on `PYTHONPATH`, so contributor guidance and verification commands now use the branch-accurate `PYTHONPATH=src` form.
- `CITATION.cff` and `paper/paper.bib` are conservative starting points and may need maintainer refinement for author list, preferred citation, and manuscript wording before submission.
