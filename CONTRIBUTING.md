# Contributing to QuMET

Thank you for your interest in contributing to QuMET! This document provides guidelines for contributing to the project.

## Code Style and Quality

QuMET follows strict code quality standards to ensure maintainability, readability, and consistency across the codebase.

### Formatting

We use the following tools to maintain consistent code formatting:

- **black**: Code formatter (line length: 88)
- **isort**: Import sorter (black-compatible profile)
- **ruff**: Fast Python linter

### Pre-commit Hooks

We use pre-commit hooks to automatically check code quality before commits. To set up:

```bash
pip install pre-commit
pre-commit install
```

Now the hooks will run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

### Running Formatters Manually

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Check and auto-fix linting issues
ruff check --fix .
```

### Docstring Style

QuMET uses **Google-style docstrings** for all public modules, classes, and functions.

#### Basic Structure

```python
def example_function(arg1, arg2, optional_arg=None):
    """One-line summary in imperative mood.
    
    More detailed description if needed. This can span multiple lines
    and explain the purpose, behavior, and any important notes.
    
    Args:
        arg1: Description of arg1. Type should be clear from context
            or specified as `arg1 (int): Description`.
        arg2: Description of arg2.
        optional_arg: Description of optional parameter. Defaults to None.
    
    Returns:
        Description of return value and its type.
    
    Raises:
        ValueError: When invalid input is provided.
        RuntimeError: When operation fails.
    
    Examples:
        >>> example_function(1, 2)
        3
    """
    pass
```

#### For Classes

```python
class ExampleClass:
    """One-line summary of the class.
    
    More detailed description of the class purpose and usage.
    
    Attributes:
        attr1: Description of attribute.
        attr2: Description of attribute.
    
    Examples:
        >>> obj = ExampleClass()
        >>> obj.method()
    """
    
    def __init__(self, param1):
        """Initialize ExampleClass.
        
        Args:
            param1: Description of initialization parameter.
        """
        pass
```

#### For Modules

Every Python module should have a module-level docstring:

```python
"""Module for quantum circuit operations.

This module provides utilities for constructing and manipulating
quantum circuits used in quantum machine learning models.
"""
```

### Comment Guidelines

#### Keep Comments

- **Rationale comments**: Explain *why* something is done
- **References**: Links to papers, specs, or issue numbers  
- **Non-obvious invariants**: Complex constraints or edge cases
- **TODOs**: With context and owner if possible

#### Remove Comments

- **Redundant comments**: That restate what the code does
- **Commented-out code**: Use git history instead
- **Obsolete TODOs**: Complete them or remove them

#### Examples

**Good comments:**

```python
# Use amplitude encoding to reduce circuit depth (see paper: arxiv.org/abs/...)
encoder = AmplitudeEncoder(n_qubits)

# HACK: Workaround for PennyLane issue #1234
# Remove this when fixed upstream
if pennylane.__version__ == "0.30.0":
    apply_hotfix()
```

**Bad comments (remove):**

```python
# Increment counter
counter += 1

# x = compute_something()  # old approach
y = compute_something_better()
```

### Testing

Before submitting a PR:

1. Ensure all existing tests pass
2. Add tests for new functionality
3. Run linters and formatters

```bash
# Run tests
pytest

# Check code quality
black --check .
isort --check .
ruff check .
```

### Continuous Integration

All PRs must pass CI checks:

- Code formatting (black, isort)
- Linting (ruff)
- Tests (if applicable)

The CI configuration is in `.github/workflows/lint.yml`.

## Documentation

### Building Documentation Locally

QuMET uses Sphinx with Google-style docstring support:

```bash
cd docs
make html
open _build/html/index.html  # On macOS
# or
xdg-open _build/html/index.html  # On Linux
```

### Documentation is Auto-generated

Documentation is automatically generated from docstrings using Sphinx's autodoc extension. Ensure your docstrings are complete and follow the Google style guide.

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes following the style guidelines
3. Run formatters and linters
4. Ensure tests pass
5. Submit a PR with a clear description of changes
6. Address review feedback

## Questions?

If you have questions about contributing, please open an issue on GitHub.

## License

By contributing to QuMET, you agree that your contributions will be licensed under the project's Apache 2.0 License.
