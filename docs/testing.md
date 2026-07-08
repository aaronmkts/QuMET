# QuMET Testing

QuMET uses `pytest` for automated verification.

## Install test dependencies

Install the contributor/test dependencies from a fresh checkout with:

```bash
pip install -e '.[dev]'
```

## Full Test Suite

Run the full suite from the repository root:

```bash
PYTHONPATH=src python -m pytest
```

## Focused Docs Check

The JOSS-facing docs contract is checked by:

```bash
PYTHONPATH=src python -m pytest tests/test_docs_content.py -v
```

That test verifies the README and linked docs for:

- purpose and statement of need,
- target audience,
- install and quickstart instructions,
- supported model, dataset, and task coverage,
- TOML configuration documentation, and
- testing guidance.

## Useful Focused Commands

For quick iteration on nearby public surfaces:

```bash
PYTHONPATH=src python -m pytest tests/test_docs_content.py tests/test_tools_config_load.py -v
```

Because the docs pages reference current example configs, it is also reasonable to rerun `tests/test_docs_content.py` whenever `README.md`, `docs/*.md`, or the example config layout changes.
