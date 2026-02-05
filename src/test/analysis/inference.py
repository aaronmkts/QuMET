import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "src")
)
import tempfile

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tbparse import SummaryReader

from qumet.dataset import get_dataset, get_dataset_info
from qumet.models import get_model


def main():
    # Load MNIST dataset (Real data)
    mnist = get_dataset("mnist", "train", "min-max", False, 7).data
    mnist = mnist.to(torch.float32) / 255.0  # Scale pixel values to [0, 1]
    mnist = mnist.unsqueeze(1)

    def get_grid_images(imgs, nimgs=8, nrow=8):
        grid = torchvision.utils.make_grid(
            imgs[:nimgs], normalize=True, nrow=nrow, pad_value=1
        )
        return grid

    def load_lightning_ckpt_to_unwrapped_model(checkpoint: str, model: torch.nn.Module):
        """
        Load a PyTorch Lightning checkpoint to a PyTorch model.
        """
        src_state_dict = torch.load(checkpoint)["state_dict"]
        tgt_state_dict = model.state_dict()
        new_tgt_state_dict = {}
        for k, v in src_state_dict.items():
            if "model." in k:
                possible_tgt_k = ".".join(k.split(".")[1:])
            else:
                possible_tgt_k = k
            if possible_tgt_k in tgt_state_dict:
                new_tgt_state_dict[possible_tgt_k] = v
        model.load_state_dict(state_dict=new_tgt_state_dict)
        return model

    dataset_info = get_dataset_info("mnist")

    # Loading the VAE-QWGAN + GMM model and sampling
    checkpoint_qvaegan = "../qumet_output/icassp2024/final_results/nwr/qvaegan/seed_9/software/training_ckpts/last.ckpt"
    model = get_model("apqgan", "image_generation", dataset_info)
    loaded_model_qvaegan = load_lightning_ckpt_to_unwrapped_model(
        checkpoint_qvaegan, model
    )

    # Sample from VAE-QWGAN + GMM
    N = 8
    gmm = joblib.load("gmm_model_seed9_mnist.pkl")
    disp_prior, _ = gmm.sample(N)
    disp_prior = torch.tensor(disp_prior, dtype=torch.float32)
    with torch.no_grad():
        image_qvaegan = loaded_model_qvaegan(disp_prior)

    # Loading the PQWGAN with uniform prior and sampling
    checkpoint_pqwgan_u = "../qumet_output/icassp2024/final_results/nwr/uniform_prior/seed_9/software/training_ckpts/last.ckpt"
    model = get_model("pqwgan_qc", "image_generation", dataset_info)
    loaded_model_pqwgan_u = load_lightning_ckpt_to_unwrapped_model(
        checkpoint_pqwgan_u, model
    )

    uniform_prior = torch.rand((N, disp_prior.shape[1]))  # Uniform sampling in [0, 1]
    with torch.no_grad():
        image_pqwgan_u = loaded_model_pqwgan_u(uniform_prior)

    # Loading the PQWGAN with Gaussian prior and sampling
    checkpoint_pqwgan_g = "../qumet_output/icassp2024/final_results/nwr/gaussian_prior/seed_9/software/training_ckpts/last.ckpt"
    model = get_model("pqwgan_qc", "image_generation", dataset_info)
    loaded_model_pqwgan_g = load_lightning_ckpt_to_unwrapped_model(
        checkpoint_pqwgan_g, model
    )

    gaussian_prior = torch.randn((N, disp_prior.shape[1]))  # Gaussian sampling
    with torch.no_grad():
        image_pqwgan_g = loaded_model_pqwgan_g(gaussian_prior)

    # Convert all images to grids
    grid_real = get_grid_images(mnist, nimgs=N, nrow=N)  # Real MNIST data
    grid_qvaegan = get_grid_images(image_qvaegan, nimgs=N, nrow=N)
    grid_pqwgan_u = get_grid_images(image_pqwgan_u, nimgs=N, nrow=N)
    grid_pqwgan_g = get_grid_images(image_pqwgan_g, nimgs=N, nrow=N)

    # Plot the images in a grid
    fig, axs = plt.subplots(4, 1, figsize=(4, 5))  # Adjust figsize to fit the extra row

    # Plot real dataset images
    axs[0].imshow(grid_real.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axs[0].set_title("Real Data", fontsize=30)
    axs[0].axis("off")

    # Plot VAE-QWGAN + GMM samples
    axs[1].imshow(grid_qvaegan.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axs[1].set_title(r"$\text{VAE-QWGAN \+} \text{GMM}(\mu,\Sigma)$", fontsize=30)
    axs[1].axis("off")

    # Plot PQWGAN with Uniform prior
    axs[2].imshow(grid_pqwgan_u.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axs[2].set_title(r"Uniform $U_{[0,1)}$", fontsize=30)
    axs[2].axis("off")

    # Plot PQWGAN with Gaussian prior
    axs[3].imshow(grid_pqwgan_g.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    axs[3].set_title(r"Gaussian $\mathcal{N}(0, \mathbb{I})$", fontsize=30)
    axs[3].axis("off")

    # Adjust spacing between subplots and remove extra space around the figure
    plt.subplots_adjust(
        wspace=0, hspace=0.1, top=0.95, bottom=0.05, left=0.01, right=0.99
    )

    # Show the plot with trimmed whitespace
    plt.show()


if __name__ == "__main__":
    main()
