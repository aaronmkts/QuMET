# JOSS Workstream 4 Report

## Internal Plan

1. Inspect the existing README, `docs/basics/*`, test layout, CLI/config loading code, and example TOML files to ground the rewrite in current repository behavior.
2. Strengthen the focused docs-content test so it rejects documented underscore CLI flags when the parser only defines hyphenated names, then run it to capture a failing RED state.
3. Strengthen the same docs test so the supported model, dataset, and task lists in JOSS-facing Markdown are checked against `qumet.models`, `AVAILABLE_DATASETS`, and `TASKS`.
4. Rewrite `README.md` and the linked docs examples so the published commands use the CLI-defined flag spellings and the supported-name lists stay aligned with the code registries.
5. Run the focused docs tests and broader verification with `/home/aaron/miniconda3/envs/qumet/bin/python`, record RED/GREEN evidence, commit the docs changes, and summarize concerns.

## TDD Evidence

### RED

Command:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_docs_content.py -v
```

Relevant output:

```text
collected 5 items

tests/test_docs_content.py::test_readme_covers_joss_public_surface PASSED
tests/test_docs_content.py::test_detailed_docs_have_expected_structure PASSED
tests/test_docs_content.py::test_public_docs_reference_existing_compatible_examples PASSED
tests/test_docs_content.py::test_public_docs_use_valid_cli_option_names FAILED
tests/test_docs_content.py::test_supported_name_lists_match_current_registries PASSED

AssertionError: Public docs use invalid CLI option spellings; use the CLI-defined hyphenated flags instead: {'README.md': ['--max_epochs'], 'docs/quickstart.md': ['--learning_rate', '--max_epochs']}
```

### GREEN

Focused docs verification:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_docs_content.py -v
```

Relevant output:

```text
collected 5 items

tests/test_docs_content.py::test_readme_covers_joss_public_surface PASSED
tests/test_docs_content.py::test_detailed_docs_have_expected_structure PASSED
tests/test_docs_content.py::test_public_docs_reference_existing_compatible_examples PASSED
tests/test_docs_content.py::test_public_docs_use_valid_cli_option_names PASSED
tests/test_docs_content.py::test_supported_name_lists_match_current_registries PASSED

============================== 5 passed in 8.88s ===============================
```

Broader verification:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest
```

Relevant output:

```text
collected 159 items
...
======================= 159 passed, 2 warnings in 9.41s ========================
```

Note: this branch still expects `PYTHONPATH=src` when running tests from the source checkout, so the public testing docs now show that command explicitly and pair it with `pip install -e '.[dev]'`.

## Files Changed

- `README.md`
- `docs/quickstart.md`
- `tests/test_docs_content.py`
- `JOSS_WORKSTREAM_REPORT.md`

## Commits Created

- `docs: fix remaining JOSS public docs review findings`

## Verification Commands

- `PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_docs_content.py -v`
- `PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest`

## Concerns

- The docs-content test now checks supported-name lists by parsing the Markdown sections rather than matching raw prose, which keeps the check pragmatic while still catching drift in the public JOSS surface.
