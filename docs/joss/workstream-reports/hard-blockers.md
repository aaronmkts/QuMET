# JOSS Workstream 1 Report

## Internal Plan

1. Inspect the packaging metadata, CLI version path, legacy `src/qmt` launcher, and the broken `bars_and_stripes` module to confirm the smallest viable fix surface.
2. Add regression tests first for:
   - Apache-2.0 package metadata and `qmt` console-script declaration in `pyproject.toml`
   - aligned package and CLI version reporting
   - importability and deterministic output of `src/qumet/dataset/manual/bitstrings/bars_and_stripes.py`
3. Run the new focused tests and capture the RED failures caused by the current broken behavior.
4. Implement the minimal source and metadata fixes inside the approved ownership scope.
5. Re-run the focused tests to capture GREEN evidence, then run `compileall` for `src/qumet` and the full test suite with `/home/aaron/miniconda3/envs/qumet/bin/python`.
6. Summarize files changed, verification evidence, commit(s), and any residual concerns in this report.

## TDD Evidence

### RED

Command:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_cli.py tests/test_dataset_bitstrings.py
```

Relevant output:

```text
FAILED tests/test_cli.py::TestQuMETCLI::test_version_constant
E   AssertionError: assert '00.00.0' == '0.1.0'

FAILED tests/test_cli.py::test_show_version_prints_package_version
E   assert '0.1.0' in "... VERSION 00.00.0 ..."

FAILED tests/test_cli.py::test_pyproject_declares_qmt_console_script_and_apache_license
E   AssertionError: assert 'MIT' == 'Apache-2.0'

FAILED tests/test_dataset_bitstrings.py::test_bars_and_stripes_module_is_importable_and_generates_unique_patterns
E   IndentationError: unexpected indent
```

### GREEN

Focused regression command:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_cli.py tests/test_dataset_bitstrings.py
```

Relevant output:

```text
============================== 27 passed in 9.84s ==============================
```

Installed console script smoke test:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pip install . --prefix /tmp/qumet-joss-install --no-build-isolation --no-deps
PYTHONPATH=/tmp/qumet-joss-install/lib/python3.11/site-packages /tmp/qumet-joss-install/bin/qmt -V
```

Relevant output:

```text
Successfully installed qumet-0.1.0
...
QuMET (qmt): QuMET's Command Line Interface
VERSION 0.1.0
```

## Files Changed

- `pyproject.toml`
- `src/qmt`
- `src/qumet/cli.py`
- `src/qumet/dataset/manual/bitstrings/bars_and_stripes.py`
- `tests/test_cli.py`
- `tests/test_dataset_bitstrings.py`
- `JOSS_WORKSTREAM_REPORT.md`

## Commits Created

- `fix: resolve JOSS hard blockers metadata and BAS helper`

## Verification Commands

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_cli.py tests/test_dataset_bitstrings.py
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m compileall src/qumet
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest
/home/aaron/miniconda3/envs/qumet/bin/python -m pip install . --prefix /tmp/qumet-joss-install --no-build-isolation --no-deps
PYTHONPATH=/tmp/qumet-joss-install/lib/python3.11/site-packages /tmp/qumet-joss-install/bin/qmt -V
```

Relevant output:

```text
compileall: exit 0 for src/qumet
pytest full suite: 157 passed, 2 warnings in 9.57s
```

## Concerns

None beyond two pre-existing CUDA/NVML warnings emitted by the full test suite.
