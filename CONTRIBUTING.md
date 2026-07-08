# Contributing to QuMET

Thanks for contributing to QuMET. This repository is aimed at reproducible development of quantum generative modelling workflows, so small, reviewable changes with tests are preferred.

## Development Setup

1. Create or update the project environment:

   ```bash
   conda env create -f environment.yml
   conda activate qumet
   ```

2. Install the package in editable mode with developer extras:

   ```bash
   python -m pip install -e ".[dev,analysis]"
   ```

3. Verify that the CLI imports:

   ```bash
   python -m qumet.cli --help
   ```

## Running Tests

Run the full test suite before opening a pull request:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest
```

For narrower iterations, run only the tests related to your change:

```bash
PYTHONPATH=src /home/aaron/miniconda3/envs/qumet/bin/python -m pytest tests/test_project_scaffold_metadata.py -v
```

The commands above are the branch-accurate default for this scaffold because the source tree lives under `src/`. If you prefer `python -m pytest`, first verify that your current environment can already import `qumet` without setting `PYTHONPATH`.

## Reporting Issues

Use [GitHub Issues](https://github.com/aaronmkts/QuMET/issues) for reproducible bug reports. Include:

- the command or API entrypoint you ran,
- the config or dataset involved,
- the full traceback or failing assertion,
- the Python and dependency environment if the failure is setup-dependent.

## Pull Requests

- Keep each pull request scoped to one change set.
- Add or update tests when behavior changes.
- Update user-facing documentation when commands, configuration, or workflows change.
- Link the relevant issue or explain the motivation in the pull request description.

## Code Style

- Follow the existing package layout under `src/qumet`.
- Use `black` formatting defaults from `pyproject.toml`.
- Prefer small functions, explicit names, and focused tests over broad refactors.
- Do not rewrite unrelated files as part of a targeted fix.
