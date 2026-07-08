# JOSS Workstream 2 Report

## Internal Plan

1. Inspect the current packaging and pytest configuration to confirm why `python -m pytest -q` from the repository root does not import `qumet` from `src/`.
2. Demonstrate the baseline failure with `/home/aaron/miniconda3/envs/qumet/bin/python -m pytest -q` and capture the relevant import error output in this report.
3. Add the smallest repo-owned test/configuration change that makes source-checkout test execution work without exporting `PYTHONPATH`, following a strict RED/GREEN cycle.
4. Add or update smoke tests around package importability and installed package metadata only where they directly support reviewer install/test confidence.
5. Verify editable-install metadata preparation with `/home/aaron/miniconda3/envs/qumet/bin/python -m pip install -e . --no-deps --dry-run --no-build-isolation` if the local environment allows it without network access.
6. Run the full test suite with `/home/aaron/miniconda3/envs/qumet/bin/python`, update this report with exact commands and outputs, review the diff, and create a commit on `joss/install-test-clean`.

## Evidence

### Baseline Failure

Command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest -q
```

Exit code: `2`

Relevant output:

```text
collected 0 items / 12 errors
E   ModuleNotFoundError: No module named 'qumet'
!!!!!!!!!!!!!!!!!!! Interrupted: 12 errors during collection !!!!!!!!!!!!!!!!!!!
```

### TDD RED

Added `tests/test_install_smoke.py` before the configuration fix.

Command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest -q tests/test_install_smoke.py
```

Exit code: `2`

Relevant output:

```text
collected 0 items / 1 error
tests/test_install_smoke.py:6: in <module>
    import qumet
E   ModuleNotFoundError: No module named 'qumet'
```

### Minimal Fix

Changed `pyproject.toml` to add:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

This keeps the source checkout runnable from the repository root without requiring a shell-exported `PYTHONPATH`.

### TDD GREEN

Command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest -q tests/test_install_smoke.py
```

Exit code: `0`

Relevant output:

```text
collected 2 items
tests/test_install_smoke.py ..                                           [100%]
============================== 2 passed in 2.27s ===============================
```

### Editable Install Metadata Check

Command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pip install -e . --no-deps --dry-run --no-build-isolation
```

Exit code: `0`

Relevant output:

```text
Obtaining file:///home/aaron/projects/QuMET-worktrees/joss-install-test
Preparing editable metadata (pyproject.toml): finished with status 'done'
Would install qumet-0.1.0
```

### Full Suite Verification

Command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest -q
```

Exit code: `0`

Relevant output:

```text
collected 156 items
tests/test_install_smoke.py ..                                           [ 43%]
======================= 156 passed, 2 warnings in 10.04s =======================
```

### Files Changed

- `pyproject.toml`
- `tests/test_install_smoke.py`
- `JOSS_WORKSTREAM_REPORT.md`

### Commits Created

- `23a37ea` `test: make source checkout pytest import qumet`

### Verification Commands

- `/home/aaron/miniconda3/envs/qumet/bin/python -m pytest -q`
- `/home/aaron/miniconda3/envs/qumet/bin/python -m pytest -q tests/test_install_smoke.py`
- `/home/aaron/miniconda3/envs/qumet/bin/python -m pip install -e . --no-deps --dry-run --no-build-isolation`

### Concerns

- Full test suite emits 2 pre-existing `torch.cuda` NVML warnings in this environment, but all tests pass.
