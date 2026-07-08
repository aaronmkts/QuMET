# JOSS Workstream 4 Report

## Internal Plan

1. Inspect the existing README, `docs/basics/*`, test layout, CLI/config loading code, and example TOML files to ground the rewrite in current repository behavior.
2. Add a focused docs-content test that encodes the JOSS readiness requirements for the README and linked documentation, then run it to capture a failing RED state.
3. Rewrite `README.md` to state QuMET's purpose clearly, add statement of need, audience, installation, quickstart, supported models/datasets/tasks, testing instructions, and links to deeper docs.
4. Add or update detailed Markdown docs for quickstart, TOML configuration structure, model/dataset overview, and testing, reusing `docs/basics/*` where helpful and avoiding community/JOSS scaffolding files owned by another workstream.
5. Run the focused docs tests and broader verification with `/home/aaron/miniconda3/envs/qumet/bin/python`, record RED/GREEN evidence, commit the docs changes, and summarize concerns.

## TDD Evidence

### RED

Command:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_docs_content.py -v
```

Relevant output:

```text
collected 3 items

tests/test_docs_content.py::test_readme_covers_joss_public_surface FAILED
tests/test_docs_content.py::test_joss_readme_links_to_detail_docs FAILED
tests/test_docs_content.py::test_detailed_docs_exist_and_cover_current_examples FAILED

AssertionError: README is missing required public-facing content: ['unified torch/pennylane framework', 'comparing qgans and qcbms from the literature', 'statement of need', 'target audience', 'pip install -e .', 'quickstart', 'supported models', 'supported datasets', 'supported tasks', 'pytest']
AssertionError: README is missing links to detailed docs: ['docs/quickstart.md', 'docs/configuration.md', 'docs/models-and-datasets.md', 'docs/testing.md']
AssertionError: Expected detailed docs are missing: ['docs/quickstart.md', 'docs/configuration.md', 'docs/models-and-datasets.md', 'docs/testing.md']
```

### GREEN

Focused docs verification:

```bash
/home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_docs_content.py -v
```

Relevant output:

```text
collected 3 items

tests/test_docs_content.py::test_readme_covers_joss_public_surface PASSED
tests/test_docs_content.py::test_joss_readme_links_to_detail_docs PASSED
tests/test_docs_content.py::test_detailed_docs_exist_and_cover_current_examples PASSED

============================== 3 passed in 0.01s ===============================
```

Broader verification:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest
```

Relevant output:

```text
collected 157 items
...
======================= 157 passed, 2 warnings in 9.23s ========================
```

Note: running `python -m pytest` without `PYTHONPATH=src` fails during collection with `ModuleNotFoundError: No module named 'qumet'`. The suite passes once run with the repository's `src` layout on `PYTHONPATH`.

## Files Changed

- `README.md`
- `docs/basics/Getting-Started.md`
- `docs/basics/Getting-Started-Using-Conda.md`
- `docs/quickstart.md`
- `docs/configuration.md`
- `docs/models-and-datasets.md`
- `docs/testing.md`
- `tests/test_docs_content.py`
- `JOSS_WORKSTREAM_REPORT.md`

## Commits Created

- `docs: refresh JOSS-facing public documentation`

## Verification Commands

- `/home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_docs_content.py -v`
- `/home/aaron/miniconda3/envs/qumet/bin/python -m pytest`
- `PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest`

## Concerns

- The repository test invocation still depends on `PYTHONPATH=src`; editable install alone was not exercised in this workstream because the task scope was docs-only.
