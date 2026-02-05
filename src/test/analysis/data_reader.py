import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..")
)
import tempfile

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tbparse import SummaryReader


def main():

    # Define the metric to plot by assigning the column name
    column_wass = (
        "wasserstein_distance"  # Change this variable to plot different metrics
    )

    log_dir_gauss = "../qumet_output/icassp2024/final_results/nwr/gaussian_prior"
    log_dir_qvaegan = "../qumet_output/icassp2024/final_results/nwr/qvaegan"
    log_dir_uniform = "../qumet_output/icassp2024/final_results/nwr/uniform_prior"

    def process_model_data(log_dir, column, dropna=True):
        df = SummaryReader(log_dir, pivot=True, extra_columns={"dir_name"}).scalars

        # Load data for seeds 19, 32, 9, 23, 42
        seed19 = df[
            df["dir_name"] == "seed_19/software/tensorboard/lightning_logs/version_0"
        ]
        seed32 = df[
            df["dir_name"] == "seed_32/software/tensorboard/lightning_logs/version_0"
        ]
        seed9 = df[
            df["dir_name"] == "seed_9/software/tensorboard/lightning_logs/version_0"
        ]
        seed23 = df[
            df["dir_name"] == "seed_23/software/tensorboard/lightning_logs/version_0"
        ]
        seed42 = df[df["dir_name"] == "software/tensorboard/lightning_logs/version_0"]

        # Merge the data for all seeds on the 'step' column
        merged_df = pd.merge(
            seed19[["step", column]],
            seed32[["step", column]],
            on="step",
            suffixes=("_19", "_32"),
        )
        merged_df = pd.merge(merged_df, seed9[["step", column]], on="step")
        merged_df.rename(columns={column: f"{column}_9"}, inplace=True)

        merged_df = pd.merge(merged_df, seed23[["step", column]], on="step")
        merged_df.rename(columns={column: f"{column}_23"}, inplace=True)

        merged_df = pd.merge(merged_df, seed42[["step", column]], on="step")
        merged_df.rename(columns={column: f"{column}_42"}, inplace=True)

        # Compute mean and std across all seeds
        merged_df[f"mean_{column}"] = merged_df[
            [
                f"{column}_19",
                f"{column}_32",
                f"{column}_9",
                f"{column}_23",
                f"{column}_42",
            ]
        ].mean(axis=1)
        merged_df[f"std_{column}"] = merged_df[
            [
                f"{column}_19",
                f"{column}_32",
                f"{column}_9",
                f"{column}_23",
                f"{column}_42",
            ]
        ].std(axis=1)

        # Handle missing data (drop NaN rows if required)
        if dropna:
            merged_df = merged_df.dropna(subset=[f"std_{column}"])
            merged_df = merged_df.dropna(subset=[f"mean_{column}"])
            merged_df["step"] = range(0, len(merged_df))

        return merged_df

    # Process data for each model
    qvaegan_merged_df_wasss = process_model_data(
        log_dir_qvaegan, column_wass, dropna=False
    )
    gaussian_merged_df_wasss = process_model_data(
        log_dir_gauss, column_wass, dropna=False
    )
    uniform_merged_df_wasss = process_model_data(
        log_dir_uniform, column_wass, dropna=False
    )

    # Define labels
    main_xlabel = "Iterations"
    main_ylabel = "Wasserstein distance"

    # Create main plot
    fig, ax_main = plt.subplots(figsize=(12, 8))

    gauss_color = "#a1dab4"
    uniform_color = "#41b6c4"
    qvaegan_color = "#225ea8"

    # Plot Gaussian prior data
    ax_main.plot(
        gaussian_merged_df_wasss["step"],
        gaussian_merged_df_wasss[f"mean_{column_wass}"],
        color=gauss_color,
        label=r"Gaussian $\mathcal{N}(0, \mathbb{I})$",
    )
    ax_main.fill_between(
        gaussian_merged_df_wasss["step"],
        gaussian_merged_df_wasss[f"mean_{column_wass}"]
        - gaussian_merged_df_wasss[f"std_{column_wass}"],
        gaussian_merged_df_wasss[f"mean_{column_wass}"]
        + gaussian_merged_df_wasss[f"std_{column_wass}"],
        color=gauss_color,
        alpha=0.3,
    )

    # Plot Uniform prior data
    ax_main.plot(
        uniform_merged_df_wasss["step"],
        uniform_merged_df_wasss[f"mean_{column_wass}"],
        color=uniform_color,
        alpha=0.8,
        label=r"Uniform $U_{[0,1)}$",
    )
    ax_main.fill_between(
        uniform_merged_df_wasss["step"],
        uniform_merged_df_wasss[f"mean_{column_wass}"]
        - uniform_merged_df_wasss[f"std_{column_wass}"],
        uniform_merged_df_wasss[f"mean_{column_wass}"]
        + uniform_merged_df_wasss[f"std_{column_wass}"],
        color=uniform_color,
        alpha=0.3,
    )

    # Plot VAE-QWGAN data
    ax_main.plot(
        qvaegan_merged_df_wasss["step"],
        qvaegan_merged_df_wasss[f"mean_{column_wass}"],
        color=qvaegan_color,
        alpha=0.8,
        label=r"VAE-QWGAN $q_{\omega}(z|x)$",
    )
    ax_main.fill_between(
        qvaegan_merged_df_wasss["step"],
        qvaegan_merged_df_wasss[f"mean_{column_wass}"]
        - qvaegan_merged_df_wasss[f"std_{column_wass}"],
        qvaegan_merged_df_wasss[f"mean_{column_wass}"]
        + qvaegan_merged_df_wasss[f"std_{column_wass}"],
        color=qvaegan_color,
        alpha=0.3,
    )

    # Set labels and customize the main plot
    ax_main.set_xlabel(main_xlabel, fontsize=30)
    ax_main.set_ylabel(main_ylabel, fontsize=30)
    ax_main.tick_params(axis="both", which="major", labelsize=30)

    # Remove the top and right borders
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["bottom"].set_linewidth(1.5)
    ax_main.spines["left"].set_linewidth(1.5)

    # Add a legend
    ax_main.legend(fontsize=30, loc="upper right")

    # Show the combined plot
    plt.show()


if __name__ == "__main__":
    main()
