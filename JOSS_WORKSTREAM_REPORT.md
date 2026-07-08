# JOSS Workstream 3 Report

## Internal plan

1. Inspect `pyproject.toml` and the current test layout to derive the correct GitHub Actions install and test commands from repo metadata instead of inventing a parallel path.
2. Add `.github/workflows/ci.yml` with triggers for `pull_request` and pushes to `master` and `main`, pinned to Python 3.11.
3. Use `actions/setup-python` with pip caching keyed from dependency files if straightforward, then install the package in editable mode with dev dependencies via `pip install -e .[dev]`.
4. Run the requested CI steps in order: import smoke check, `python -m compileall -q src/qumet`, and `python -m pytest -q`.
5. Validate the workflow locally as far as this branch allows, without masking the known compile-time syntax blocker that another workstream is fixing.
6. Update this report with files changed, verification evidence, concerns, and the commit created.

## Constraints and notes

- Scope is limited to CI workflow work for JOSS readiness.
- I will not fix existing package or documentation blockers in this worktree.
- TDD is not directly applicable unless I add executable validation code or tests; for a pure workflow YAML change I will record that limitation and the validation performed instead.
- The known `compileall` failure on this base branch must remain visible in CI.

## Files changed

- Added `.github/workflows/ci.yml` to run editable install, import smoke, `compileall`, and `pytest` on `pull_request` and pushes to `master`/`main`.
- Added `JOSS_WORKSTREAM_REPORT.md` to capture the workstream plan, validation evidence, and concerns.

## TDD evidence

- TDD was not applicable for the implemented change because the scoped deliverable is a GitHub Actions workflow definition only.
- I did not add executable repository code or tests, so there was no meaningful red-green cycle to run inside the repo for this workstream.
- Validation focused on workflow structure and branch-aware command outcomes instead.

## Verification commands

- `sed -n '1,220p' .github/workflows/ci.yml`
  - Confirmed the workflow file exists with the intended triggers and steps.
- `python3 -c "import pathlib, yaml; data = yaml.load(pathlib.Path('.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader); print(list(data.keys()))"`
  - Passed, printing `['name', 'on', 'jobs']`, which is a reasonable structural check for the YAML document.
- `python3 -m compileall -q src/qumet`
  - Failed as expected on the pre-existing blocker: `IndentationError: unexpected indent (bars_and_stripes.py, line 1)`.
  - Also surfaced existing `SyntaxWarning` messages in unrelated files; these were left untouched because this workstream does not own them.
- `python3 -m pytest -q`
  - Could not run in the current shell because `pytest` is not installed locally: `/usr/bin/python3: No module named pytest`.
- `python3 -c "import sys; sys.path.insert(0, 'src'); import qumet; print(qumet.__file__)"`
  - Could not complete in the current shell because runtime dependencies are not installed locally: `ModuleNotFoundError: No module named 'torch'`.

## Commits created

- Planned commit: `ci: add GitHub Actions workflow for JOSS checks`

## Concerns

- The workflow is expected to fail on this base branch until the separate syntax-fix workstream lands, because `python -m compileall -q src/qumet` correctly exposes the existing `bars_and_stripes.py` indentation error.
- I validated YAML structure locally, but I did not add a repository-local workflow test harness because that would add maintenance surface without improving confidence enough for this scoped YAML-only change.
- Full local execution of the CI command sequence was not possible in the current shell because the environment lacks installed package/test dependencies.
