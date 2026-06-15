# QuMET Documentation Guide

This document explains how to build, maintain, and deploy QuMET's documentation.

## Overview

QuMET uses [Sphinx](https://www.sphinx-doc.org/) to generate documentation from:
- Google-style docstrings in the source code
- ReStructuredText (`.rst`) files in `docs/source/`
- Markdown (`.md`) files (via MyST parser)

The documentation is automatically built and hosted on [Read the Docs](https://readthedocs.org/).

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated HTML will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser.

### Other Build Formats

```bash
# PDF documentation
make latexpdf

# EPUB documentation
make epub

# Clean build artifacts
make clean
```

## Read the Docs Configuration

### Setup

The repository is configured to automatically build documentation on Read the Docs:

1. **Configuration File**: `.readthedocs.yaml` in the repository root
2. **Python Version**: Python 3.11
3. **Build Requirements**: 
   - Main dependencies from `requirements.txt`
   - Documentation dependencies from `docs/requirements.txt`

### Accessing Documentation

Once configured, documentation is available at:
```
https://qumet.readthedocs.io/
```

### Triggering Builds

Documentation builds are triggered automatically:
- On every push to the `main` branch
- On every pull request (preview builds)
- Manual builds can be triggered from the Read the Docs dashboard

### Build Configuration

The `.readthedocs.yaml` file specifies:
- Ubuntu 22.04 build environment
- Python 3.11
- Sphinx configuration location
- Output formats (HTML, PDF, EPUB)

## Documentation Structure

```
docs/
├── Makefile              # Build commands
├── requirements.txt      # Sphinx and extensions
├── source/
│   ├── conf.py          # Sphinx configuration
│   ├── index.rst        # Main documentation page
│   ├── getting_started.rst
│   └── api/             # API reference
│       ├── index.rst
│       ├── models.rst
│       ├── dataset.rst
│       └── ...
├── build/               # Generated documentation (gitignored)
├── basics/              # Getting started guides
└── labs/                # Jupyter notebook examples
```

## Writing Documentation

### Module Docstrings

Add a docstring at the top of every Python module:

```python
"""Module for quantum circuit operations.

This module provides utilities for constructing and manipulating
quantum circuits used in quantum machine learning models.
"""
```

### Class Docstrings

```python
class QuantumCircuit:
    """Represents a quantum circuit.
    
    This class provides methods to build and manipulate quantum circuits
    for machine learning applications.
    
    Attributes:
        n_qubits: Number of qubits in the circuit.
        depth: Circuit depth (number of layers).
    
    Examples:
        >>> circuit = QuantumCircuit(n_qubits=4)
        >>> circuit.add_layer()
    """
```

### Function Docstrings

```python
def apply_rotation(angle, qubit):
    """Apply rotation gate to a qubit.
    
    Args:
        angle: Rotation angle in radians.
        qubit: Target qubit index.
    
    Returns:
        Updated quantum state.
    
    Raises:
        ValueError: If qubit index is out of range.
    """
```

### Adding New Documentation Pages

1. Create a new `.rst` file in `docs/source/`
2. Add it to a `toctree` directive in an existing file
3. Build and preview locally

Example:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   tutorials/basics
   tutorials/advanced
```

## Sphinx Extensions

QuMET documentation uses:

- `sphinx.ext.autodoc`: Generate documentation from docstrings
- `sphinx.ext.napoleon`: Support Google-style docstrings
- `sphinx.ext.viewcode`: Add links to source code
- `sphinx.ext.intersphinx`: Link to external documentation (PyTorch, NumPy, etc.)
- `myst_parser`: Support Markdown files

## Troubleshooting

### Import Errors

If Sphinx can't import modules:
- Ensure `sys.path` is correctly set in `docs/source/conf.py`
- Check that dependencies are installed
- Try building with `--no-cache-dir` flag

### Build Warnings

Sphinx treats warnings as errors in CI. Common issues:
- Missing docstrings for public functions
- Broken cross-references
- Invalid ReStructuredText syntax

### Read the Docs Build Failures

Check the build log on Read the Docs dashboard:
1. Go to your Read the Docs project page
2. Click "Builds"
3. Review the failing build log
4. Fix issues in `.readthedocs.yaml` or `docs/source/conf.py`

## Updating Documentation

### After Code Changes

If you modify function signatures or behavior:
1. Update the docstrings
2. Rebuild documentation locally
3. Verify the changes are reflected

### After Adding New Modules

1. Add corresponding `.rst` file in `docs/source/api/`
2. Add to the `toctree` in `docs/source/api/index.rst`
3. Rebuild and verify

## Continuous Integration

Documentation quality is enforced via:
- Pre-commit hooks check docstring presence
- CI builds documentation and fails on warnings
- Read the Docs builds preview documentation for pull requests

## Best Practices

1. **Write docstrings as you code**: Don't leave documentation for later
2. **Use examples**: Code examples in docstrings are incredibly helpful
3. **Link to related functions**: Use `:func:`, `:class:`, `:mod:` roles
4. **Keep it up-to-date**: Update docs when changing code
5. **Build locally**: Always preview documentation before pushing

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Read the Docs Documentation](https://docs.readthedocs.io/)
- [ReStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
