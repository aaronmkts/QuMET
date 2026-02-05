import os
import sys

import numpy as np
import pandas as pd

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "src")
)
import re

import wandb

from qumet.tools.checkpoint_load import *


def split_seed_from_name(run_name: str):
    """
    Splits off the '_seedXX' and optional '_TLv1' parts of a run name, if present.
    Returns (experiment_group, seed_number).
    E.g. 'vaeqwgan_mnist01_seed42' -> ('vaeqwgan_mnist01', '42')
         'pqwgan_mnist01_gaussian_seed69' -> ('pqwgan_mnist01_gaussian', '69')
         'vaeqwgan_mnist017_seed42_TLv1' -> ('vaeqwgan_mnist017_TL', '42')
    If no seed found, returns (run_name, None).
    """
    pattern = r"^(.*)_seed(\d+)(_TLv\d+)?$"
    match = re.match(pattern, run_name)
    if match:
        group = match.group(1)  # everything before _seed
        seed_num = match.group(2)
        if match.group(3):  # if '_TLvX' is present
            group += "_TL"
        return group, seed_num
    else:
        return run_name, None


# -------------------------------
# Load W&B runs and process history
# -------------------------------
api = wandb.Api()
columns_to_ignore = [
    "TSNE/epoch_12",
    "TSNE/epoch_4",
    "TSNE/epoch_7",
    "TSNE/epoch_14",
    "TSNE/epoch_1",
    "TSNE/epoch_10",
    "TSNE/epoch_11",
    "TSNE/epoch_6",
    "TSNE/epoch_5",
    "TSNE/epoch_9",
    "TSNE/epoch_8",
    "TSNE/epoch_2",
    "TSNE/epoch_13",
    "TSNE/epoch_0",
    "TSNE/epoch_3",
    "images/real",
    "images/sample",
    "images/recon",
]

# Query runs from the W&B project "qumet/QMI-ModeCollapse"
runs = api.runs("qumet/QMI-ModeCollapse")
dfs = []
for run in runs:
    run_df = run.history(samples=5000)
    run_name = run.config.get("run_name", "unknown")
    run_df["run_name"] = run_name
    run_df["run_id"] = run.id
    dfs.append(run_df)

all_runs_df = pd.concat(dfs, ignore_index=True)
filtered_df = all_runs_df.drop(columns=columns_to_ignore, errors="ignore")

# Split run_name into experiment_group and seed.
filtered_df["experiment_group"], filtered_df["seed"] = zip(
    *filtered_df["run_name"].apply(split_seed_from_name)
)

# Sort by experiment_group, run_id, and step.
filtered_df = filtered_df.sort_values(by=["experiment_group", "run_id", "_step"])


# -------------------------------
# Assign epoch numbers per run based on when a particular metric is logged.
# -------------------------------
def assign_epoch_numbers(run_df, metric_column):
    # Drop rows with missing metric values and assign epochs sequentially.
    epoch_df = run_df.dropna(subset=[metric_column]).copy()
    epoch_df["epoch"] = np.arange(len(epoch_df))
    return epoch_df


# Here we choose one metric column to define epochs.
# (For example, "metrics/jsd" is used below.)
metric_for_epoch = "metrics/jsd"
epoch_assigned_df = filtered_df.groupby(
    ["experiment_group", "run_id"], group_keys=False
).apply(assign_epoch_numbers, metric_column=metric_for_epoch)


# -------------------------------
# Function to get metric stats
# -------------------------------
def get_metric_stats(allowed_groups, metrics, epoch):
    """
    For each metric in 'metrics', compute the mean and standard deviation at the given 'epoch'
    across all runs belonging to the allowed experiment groups.

    Parameters
    ----------
    allowed_groups : list of str
        A list of allowed experiment group names (e.g. ["vaeqwgan_mnist01", "pqwgan_mnist01_gaussian", ...])
    metrics : list of str
        A list of column names (metrics) to compute stats for.
    epoch : int
        The epoch number to filter on.

    Returns
    -------
    results : dict
        A dictionary where each key is a metric (column name) and the value is a DataFrame
        with columns: "experiment_group", "mean", and "std", computed over the runs.
    """
    results = {}
    # Filter the epoch-assigned data to only the allowed groups and the given epoch.
    df_filtered = epoch_assigned_df[
        (epoch_assigned_df["experiment_group"].isin(allowed_groups))
        & (epoch_assigned_df["epoch"] == epoch)
    ]
    for metric in metrics:
        # Group by experiment_group and compute the mean and std for the metric.
        agg_df = (
            df_filtered.groupby("experiment_group")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        results[metric] = agg_df
    return results


# -------------------------------
# Example usage:
# -------------------------------
if __name__ == "__main__":
    # Define the allowed experiment groups.
    allowed_groups = [
        "pqwgan_fmnist017_gaussian",
        "pqwgan_fmnist017_uniform",
    ]

    # Specify the list of metric columns you want to aggregate.
    # For example, compute stats at epoch 1 for "metrics/jsd" and "metrics/ndb_k".
    metrics_to_check = [
        "metrics/jsd",
        "metrics/ndb_k",
        "metrics/ssim",
        "metrics/psnr",
        "metrics/fid",
        "metrics/cos_sim",
    ]
    epoch_to_check = 15  # Change this to the desired epoch (e.g., 15 for final epoch)

    stats = get_metric_stats(allowed_groups, metrics_to_check, epoch_to_check)

    # Print the results.
    for metric, agg_df in stats.items():
        print(f"\nMetric: {metric} at epoch {epoch_to_check}")
        print(agg_df.to_string(index=False))
