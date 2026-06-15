import os
import sys

import matplotlib.pyplot as plt
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

column = "wasserstein_distance"
filtered_df = filtered_df.dropna(subset=[column])
filtered_df = filtered_df.sort_values(by="_step")

agg_df = (
    filtered_df.groupby(["experiment_group", "_step"], as_index=False)[column]
    .agg(["mean", "std"])
    .reset_index()
)

# Correctly normalize epochs per group
agg_df["epoch"] = 0
for group in agg_df["experiment_group"].unique():
    group_mask = agg_df["experiment_group"] == group
    max_step_group = agg_df.loc[group_mask, "_step"].max()
    agg_df.loc[group_mask, "epoch"] = (
        agg_df.loc[group_mask, "_step"] / max_step_group * 15
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
    plt.legend(fontsize=30, loc="upper right")

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
    "gan_mnist017_uniform",
]

colors = {
    "vaeqwgan_mnist017": "#225ea8",  # e.g., VAE-QWGAN
    "pqwgan_mnist017_gaussian": "#a1dab4",  # e.g., Gaussian prior
    "pqwgan_mnist017_uniform": "#41b6c4",  # If you want to add Uniform later
    "gan_mnist017_uniform": "#ff8a33",
}

group_labels = {
    "vaeqwgan_mnist017": "VAE-QWGAN",
    "pqwgan_mnist017_gaussian": r"PQWGAN+$\mathcal{N}(0, \mathbb{I})$",
    "pqwgan_mnist017_uniform": r"PQWGAN+$U_{[0, 1)}$",
    "gan_mnist017_uniform": r"GAN+$U_{[0, 1)}$",
}

plot_metric_with_std(
    allowed_groups=allowed_groups,
    metric_column="wasserstein_distance",
    agg_data=agg_df,
    colors=colors,
    group_labels=group_labels,
    xlabel="Epoch",
    ylabel="Wasserstein Distance",
)


def get_final_metrics(agg_data, allowed_groups):
    """
    For each experiment group in allowed_groups, get the row with the maximum _step,
    and return a DataFrame with the final mean, std, and epoch.
    """
    final_metrics = []
    for group in allowed_groups:
        group_data = agg_data[agg_data["experiment_group"] == group]
        if not group_data.empty:
            # Find the row with the maximum _step in this group
            final_row = group_data.loc[group_data["_step"].idxmax()]
            final_metrics.append(
                {
                    "experiment_group": group,
                    "final_epoch": final_row["epoch"],
                    "final_mean": final_row["mean"],
                    "final_std": final_row["std"],
                }
            )
    return pd.DataFrame(final_metrics)


# Using the aggregated DataFrame (agg_df) from your previous code
final_metrics_df = get_final_metrics(agg_df, allowed_groups)
print("Final metrics for each experiment group:")
print(final_metrics_df)
