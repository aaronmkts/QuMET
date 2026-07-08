# QuMET

QuMET is a unified Torch/PennyLane framework for comparing QGANs and QCBMs from the literature. It provides a single training and evaluation surface for quantum and hybrid generative models, common dataset handling, TOML-based experiment configuration, and reproducible CLI workflows for benchmarking model behavior across tasks.

![QuMET pipeline](./docs/imgs/QuMET_Pipeline.png)

## Statement of Need

Research code for quantum generative modelling is often released as paper-specific implementations with incompatible training loops, dataset preprocessing, and evaluation setups. That makes it hard to compare QGAN and QCBM results fairly, reproduce reported behavior, or extend prior work without rebuilding infrastructure each time. QuMET addresses that gap by offering a shared Torch/PennyLane framework for running literature-inspired QGAN, QCBM, GAN, and VAE baselines through one configuration and CLI interface.

## Target Audience

QuMET is aimed at researchers, research software engineers, and advanced students who need a reproducible codebase for:

- comparing quantum generative models on common datasets and tasks,
- testing new training or evaluation ideas against existing model families, and
- documenting experiment configuration in a form that can be versioned and rerun.

## Installation

QuMET requires Python 3.11 or newer.

```bash
git clone https://github.com/aaronmkts/QuMET.git
cd QuMET
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You can also install the Conda environment described in [docs/basics/Getting-Started-Using-Conda.md](docs/basics/Getting-Started-Using-Conda.md).

## Quickstart

The repository includes runnable TOML examples under `src/configs/`. A minimal discrete-generation run uses the current `qcbm` model and `bars_and_stripes` dataset:

```bash
python src/qmt train --config src/configs/by_model/qcbm/qcbm.toml
```

You can also override configuration values from the CLI. For example:

```bash
python src/qmt train qcbm bars_and_stripes --task discrete_generation --batch-size 10 --max-epochs 5 --accelerator cpu
```

See [docs/quickstart.md](docs/quickstart.md) for a slightly fuller walkthrough.

## Supported Models

QuMET currently exposes the following model names through `qumet.models` and the CLI:

- `qcbm`
- `patchgan`
- `mosaiq`
- `vaeqwgan`
- `pqwgan_qc`
- `qinr`
- `gan`
- `vae`

These cover QCBM-style discrete generators and QGAN-style or hybrid image/continuous generators implemented on top of Torch and PennyLane.

## Supported Datasets

Current dataset names include:

- Vision datasets: `mnist`, `fashion_mnist`, `cifar10`
- Manual continuous datasets: `2d_gaussian`, `2d_grid_gaussian`, `2d_ring_gaussian`
- Manual bitstring datasets: `bars_and_stripes`, `2d_gaussian_b`, `2d_grid_gaussian_b`, `2d_ring_gaussian_b`

## Supported Tasks

The CLI accepts these task names:

- `discrete_generation`
- `continuous_generation`
- `image_generation`

Model-dataset compatibility depends on the task. The `qcbm` examples target `discrete_generation`, while image models such as `pqwgan_qc` and `vae` target `image_generation`.

## Configuration

QuMET accepts flat TOML experiment files with top-level CLI keys such as `model`, `dataset`, `task`, `batch_size`, and `accelerator`, plus optional nested sections such as `[metrics]`. Example configuration files live under:

- `src/configs/by_model/`
- `src/configs/by_environment/`
- `src/configs/by_paper/`

See [docs/configuration.md](docs/configuration.md) for the TOML structure, merge rules, and example locations.

## Testing

For contributor or reviewer test runs, install the development extra first:

```bash
pip install -e '.[dev]'
```

Run the test suite from the repository root:

```bash
PYTHONPATH=src python -m pytest
```

For the docs-focused checks added for JOSS readiness:

```bash
PYTHONPATH=src python -m pytest tests/test_docs_content.py -v
```

See [docs/testing.md](docs/testing.md) for focused test commands and verification notes.

## Detailed Docs

- [Quickstart](docs/quickstart.md)
- [Configuration](docs/configuration.md)
- [Models and datasets](docs/models-and-datasets.md)
- [Testing](docs/testing.md)
- [Research memory boundary](docs/research-memory-boundary.md)

## Repository Layout

- `src/qumet/`: Python package, CLI, models, datasets, actions, and utilities
- `src/configs/`: example TOML configurations grouped by model, environment, and paper workflow
- `tests/`: pytest suite
- `docs/`: user-facing documentation and notebooks
