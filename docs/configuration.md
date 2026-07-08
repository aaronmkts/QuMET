# QuMET Configuration

QuMET uses TOML files to describe experiment settings. These files are loaded by `qumet.tools.config_load.post_parse_load_config`, merged with CLI defaults, and then overridden by any explicit CLI flags.

## Merge Order

The effective configuration order is:

1. CLI defaults in `qumet.cli.CLI_DEFAULTS`
2. top-level values loaded from a TOML config file
3. manual CLI overrides such as `--max-epochs 5`

## Core Top-Level Keys

Most example files define a flat set of top-level keys, for example:

```toml
model = "qcbm"
dataset = "bars_and_stripes"
task = "discrete_generation"
n_qubits = 9
batch_size = 10
learning_rate = 1e-1
max_epochs = 100
seed = 0
num_workers = 0
num_devices = 1
accelerator = "cpu"
project = "bars_and_stripes"
```

Common keys include:

- experiment identity: `model`, `dataset`, `project`, `seed`
- task selection: `task`, `transform`
- training controls: `training_optimizer`, `learning_rate`, `weight_decay`, `max_epochs`, `max_steps`, `batch_size`, `n_samples`
- runtime settings: `num_workers`, `num_devices`, `num_nodes`, `accelerator`, `strategy`

## Metrics Subsections

Metrics are defined under an optional `[metrics]` table with nested subtables for callback-specific arguments.

```toml
[metrics]
use_metrics = ["FIDEvaluationCallback", "NDB_JSD_EvaluationCallback", "GANImagesCallback"]

[metrics.FIDEvaluationCallback]
every_n_epochs = 1
feature = 64

[metrics.NDB_JSD_EvaluationCallback]
every_n_epochs = 1
number_of_bins = 50
z_threshold = "NA"
```

The loader also interprets `"NA"` as `None` because TOML does not encode Python `None` directly.

## Where Example Configs Live

Current example configs live in:

- `src/configs/by_model/` for model-oriented examples such as `qcbm`, `pqwgan_qc`, and `vae`
- `src/configs/by_environment/` for environment-specific settings
- `src/configs/by_paper/` for paper-style experiment groupings

Useful starting points:

- `src/configs/by_model/qcbm/qcbm.toml`
- `src/configs/by_model/qgan/pqwgan_qc.toml`

## Recommended Workflow

1. Copy the closest existing TOML example.
2. Change `model`, `dataset`, and `task` only to values supported by the current registry.
3. Run with `python src/qmt train --config path/to/config.toml`.
4. Apply small one-off CLI overrides instead of duplicating many near-identical TOML files.
