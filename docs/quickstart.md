# QuMET Quickstart

This quickstart shows the smallest current workflow for running QuMET from a fresh checkout.

## 1. Install QuMET

```bash
git clone https://github.com/aaronmkts/QuMET.git
cd QuMET
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Run a current example config

The repository ships example TOML files under `src/configs/`. A minimal discrete-generation example uses the current `qcbm` model and `bars_and_stripes` dataset:

```bash
python src/qmt train --config src/configs/by_model/qcbm/qcbm.toml
```

That config sets:

- `model = "qcbm"`
- `dataset = "bars_and_stripes"`
- `task = "discrete_generation"`
- `accelerator = "cpu"`

## 3. Override values from the CLI

CLI arguments take precedence over config values, so you can keep the same file and adjust a few settings:

```bash
python src/qmt train --config src/configs/by_model/qcbm/qcbm.toml --max_epochs 5 --seed 123 --accelerator cpu
```

You can also run without a config file when you want a direct command:

```bash
python src/qmt train qcbm bars_and_stripes --task discrete_generation --n_qubits 9 --batch-size 10 --learning_rate 0.1 --max_epochs 5 --accelerator cpu
```

## 4. Explore another current model family

For image-generation workflows, start from one of the QGAN examples such as:

- `src/configs/by_model/qgan/pqwgan_qc.toml`
- `src/configs/by_model/qgan/patchgan.toml`

These examples use names that are currently registered in the codebase, including `pqwgan_qc`, `patchgan`, `mnist`, and `fashion_mnist`.

## 5. Read the next docs

- [Configuration](configuration.md) for TOML structure and merge rules
- [Models and datasets](models-and-datasets.md) for current names and task coverage
- [Testing](testing.md) for verification commands
