# Getting Started

This page is a short pointer to the current QuMET user docs.

## Recommended entry points

- Start with the repository [README](../../README.md) for the project overview, statement of need, install command, and a minimal quickstart.
- Use [../quickstart.md](../quickstart.md) for the smallest runnable example.
- Use [../configuration.md](../configuration.md) for TOML structure and example config locations.

## Minimal install

```bash
git clone https://github.com/aaronmkts/QuMET.git
cd QuMET
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the CLI

```bash
python src/qmt train --config src/configs/by_model/qcbm/qcbm_2dgauss.toml
```
