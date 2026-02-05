import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "src")
)
import re

import torch.optim as optim
import wandb

from qumet.dataset import QuMETDataModule, get_dataset, get_dataset_info
from qumet.models import get_model, get_model_info
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


r''' 
def split_seed_from_name(run_name: str):
    """
    Splits off the last '_seedXX' part of a run name, if present.
    Returns (experiment_group, seed_number).
    E.g. 'vaeqwgan_mnist01_seed42' -> ('vaeqwgan_mnist01', '42')
         'pqwgan_mnist01_gaussian_seed69' -> ('pqwgan_mnist01_gaussian', '69')
    If no seed found, returns (run_name, None).
    """
    pattern = r'^(.*)_seed(\d+)$'
    match = re.match(pattern, run_name)
    if match:
        group = match.group(1)  # everything before _seed
        seed_num = match.group(2) 
        return group, seed_num
    else:
        # No _seed pattern
        return run_name, None
'''

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

# Corrected unpacking with new split_seed_from_name
filtered_df["experiment_group"], filtered_df["seed"] = zip(
    *filtered_df["run_name"].apply(split_seed_from_name)
)

column = "metrics/ndb_k"

# Sort by experiment group, run_id, and step
filtered_df = filtered_df.sort_values(by=["experiment_group", "run_id", "_step"])


# Function to explicitly assign epoch numbers per run
def assign_epoch_numbers(run_df, metric_column):
    epoch_df = run_df.dropna(subset=[metric_column]).copy()
    epoch_df["epoch"] = np.arange(len(epoch_df))
    return epoch_df


# Apply the epoch assignment per each run independently
epoch_assigned_df = filtered_df.groupby(
    ["experiment_group", "run_id"], group_keys=False
).apply(assign_epoch_numbers, metric_column=column)

# Group by experiment group and epoch explicitly
agg_df = (
    epoch_assigned_df.groupby(["experiment_group", "epoch"], as_index=False)[column]
    .agg(["mean", "std"])
    .reset_index()
)


def plot_metric_with_std(
    allowed_groups,
    metric_column,
    agg_data,
    colors,
    group_labels=None,
    xlabel="Epoch",
    ylabel=None,
    plot_title=None,
):

    if group_labels is None:
        group_labels = {}

    plt.figure(figsize=(12, 8))

    for group in allowed_groups:
        subset = agg_data[agg_data["experiment_group"] == group]
        if subset.empty:
            continue  # skip if no data for this group

        color = colors.get(group, None)
        label = group_labels.get(group, group)  # custom label if available

        # Plot mean vs. epoch
        plt.plot(subset["epoch"], subset["mean"], color=color, label=label, linewidth=2)
        # Fill between mean ± std
        plt.fill_between(
            subset["epoch"],
            subset["mean"] - subset["std"],
            subset["mean"] + subset["std"],
            color=color,
            alpha=0.3,
        )

    # Force x-axis to show 0..15 (epochs)
    plt.xlim(0, 15)
    # Optional: show integer ticks from 0..15
    plt.xticks(range(16))

    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(
        ylabel if ylabel else metric_column.replace("_", " ").title(), fontsize=30
    )

    plt.grid(True)
    # plt.legend(fontsize=30, loc="upper right")

    # Customize spines and tick parameters
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    plt.tick_params(axis="both", which="major", labelsize=30)

    plt.show()


allowed_groups = [
    "vaeqwgan_mnist017",
    "pqwgan_mnist017_gaussian",
    "pqwgan_mnist017_uniform",
]

colors = {
    "vaeqwgan_mnist017": "#225ea8",  # e.g., VAE-QWGAN
    "pqwgan_mnist017_gaussian": "#a1dab4",  # e.g., Gaussian prior
    "pqwgan_mnist017_uniform": "#41b6c4",  # If you want to add Uniform later
    # ff8a33 for classcical uniform
}

group_labels = {
    "vaeqwgan_mnist017": "VAE-QWGAN",
    "pqwgan_mnist017_gaussian": r"PQWGAN + $\mathcal{N}(0, \mathbb{I})$",
    "pqwgan_mnist017_uniform": r"PQWGAN + $U_{[0, 1)}$",
}

plot_metric_with_std(
    allowed_groups=allowed_groups,
    metric_column="wasserstein_distance",
    agg_data=agg_df,
    colors=colors,
    group_labels=group_labels,
    xlabel="Epoch",
    ylabel="PSNR",
)

breakpoint()
