# Getting Started Using Conda

Use this route when you prefer Conda over `venv`.

## Install

```bash
git clone https://github.com/aaronmkts/QuMET.git
cd QuMET
conda env create -f environment.yml
conda activate qumet
pip install -e .
```

## Run a current example

```bash
python src/qmt train --config src/configs/by_model/qcbm/qcbm_2dgauss.toml
```

## Next references

- [README](../../README.md)
- [../quickstart.md](../quickstart.md)
- [../configuration.md](../configuration.md)
