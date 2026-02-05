Getting Started
===============

This guide will help you install and start using QuMET.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.10 or 3.11
- pip or conda package manager

Using pip
~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/aaronmkts/QuMET.git
   cd QuMET
   
   # Install dependencies
   pip install -r requirements.txt

Using conda
~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/aaronmkts/QuMET.git
   cd QuMET
   
   # Create and activate conda environment
   conda env create -f environment.yml
   conda activate qumet

Quick Start
-----------

Training a Model
~~~~~~~~~~~~~~~~

Use the command-line interface to train a quantum generative model:

.. code-block:: bash

   python -m qumet.cli train --model qgan --dataset mnist --config configs/qgan_mnist.toml

Configuration
~~~~~~~~~~~~~

QuMET uses TOML configuration files for model and training parameters. Example configuration:

.. code-block:: toml

   [model]
   name = "qgan"
   n_qubits = 10
   
   [training]
   epochs = 100
   batch_size = 32
   learning_rate = 0.001

See the ``configs/`` directory for more examples.

Development
-----------

For development and contribution, see the :doc:`CONTRIBUTING <../CONTRIBUTING>` guide.

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Install pre-commit hooks for automatic code quality checks:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   pytest

Code Formatting
~~~~~~~~~~~~~~~

.. code-block:: bash

   black .
   isort .
   ruff check --fix .

Next Steps
----------

- Explore the :doc:`api/index` for detailed API documentation
- Check out examples in the ``docs/labs/`` directory
- Read about available models and datasets
