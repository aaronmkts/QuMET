import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import sys
import os
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"src"
     )
    )
import seaborn as sns
from qumet.dataset import QuMETDataModule
from qumet.models import get_model, get_model_info
from qumet.dataset import get_dataset, get_dataset_info
import torch.optim as optim
from qumet.tools.checkpoint_load import *
import wandb
import re


# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------
# Load dataset (assumed to return both data and labels)
dataset = get_dataset('mnist', 'train', 'min-max', False, 2600, 8)
mnist = dataset.data
real_labels = dataset.targets  # Assumes labels are provided by the dataset

mnist = mnist.to(torch.float32) / 255.0  
mnist = mnist.unsqueeze(1)          


# -------------------------------
# 2. Load the Pre-trained Model
# -------------------------------

# VAEQWGAN
APQGAN = get_model("apqgan", "image_generation", get_dataset_info("mnist"))
checkpoint_path_vae = "artifacts/model-f4m5ngxy:v1/model.ckpt"
model_vaeqwgan = load_model(checkpoint_path_vae, "pl", APQGAN)

#PQWGAN + Uniform

PQWGAN_UNIFORM = get_model("pqwgan_qc", "image_generation", get_dataset_info("mnist"))
checkpoint_path_pqwgan_uniform = "artifacts/model-r1l5d7c7:v1/model.ckpt"
model_pqwgan = load_model(checkpoint_path_pqwgan_uniform, "pl", PQWGAN_UNIFORM)
# -------------------------------

#PQWGAN + Gaussian
PQWGAN_GAUSSIAN = get_model("pqwgan_qc", "image_generation", get_dataset_info("mnist"))
checkpoint_path_pqwgan_gaussian = "artifacts/model-9i4sr047:v1/model.ckpt"
model_pqwgan_gaussian = load_model(checkpoint_path_pqwgan_gaussian, "pl", PQWGAN_GAUSSIAN)
# -------------------------------

#Classical GANS+Uniform
GAN_UNIFORM = get_model("gan", "image_generation", get_dataset_info("mnist"))
checkpoint_path_gan_uniform = "artifacts/model-tdib7ub6:v1/model.ckpt"
model_gan_uniform = load_model(checkpoint_path_gan_uniform, "pl", GAN_UNIFORM)

# -------------------------------
# 3. Generate images
# -------------------------------

# Generate images using the model
model_vaeqwgan.eval()
model_pqwgan.eval()
model_pqwgan_gaussian.eval()
model_gan_uniform.eval()



latent_z = []
model_vaeqwgan.eval()
with torch.no_grad():
    mu, log_var, z = model_vaeqwgan.vae_forward(mnist)
    latent_z.append(z)
z_samples = torch.cat(latent_z, dim=0).cpu().numpy()

n_samples = 8  
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=9).fit(z_samples)
disp_prior, _ = gmm.sample(n_samples)
disp_prior = torch.tensor(disp_prior, dtype=torch.float32)

model_vaeqwgan.eval()
with torch.no_grad():
    fake_images = model_vaeqwgan(disp_prior)
fake_images_np = fake_images.view(-1, 28, 28)


z_uniform = torch.rand(8,7)
fake_pqwgan_uniform_samples = model_pqwgan(z_uniform).view(-1, 28, 28)

z_gaussian = torch.randn(8,7)
fake_pqwgan_gaussian_samples = model_pqwgan_gaussian(z_gaussian).view(-1, 28, 28)

z_gan_uniform = torch.rand(8,7)
fake_gan_uniform_samples = model_gan_uniform(z_gan_uniform).view(-1, 28, 28)


real_images = mnist[:8].squeeze(1).detach().cpu()
vae_qwgan_images = fake_images_np.detach().cpu()
uniform_images = fake_pqwgan_uniform_samples.detach().cpu()
gaussian_images = fake_pqwgan_gaussian_samples.detach().cpu()
classical_gan_images = fake_gan_uniform_samples.detach().cpu()
import matplotlib.gridspec as gridspec
# Combine all into a list
all_samples = [
    (real_images, "Real Data"),
    (vae_qwgan_images, r"VAE-QWGAN + GMM($\mu$, $\Sigma$)"),
    (gaussian_images, r"PQWGAN + $\mathcal{N}(0, \mathbb{I})$"),
    (uniform_images, r"PQWGAN + $U_{[0, 1)}$"),
    (classical_gan_images, r"GAN + Uniform $U_{[0, 1)}$")
]

num_images = 8  # Number of images per row

# 1) Create a larger figure to make images bigger
fig = plt.figure(figsize=(16, 14))  # width x height in inches

# 2) We have 2 "rows" (title + images) per model
rows = 2 * len(all_samples)

# 3) Define height ratios:
#    - first row is for the title (smaller),
#    - second row is for the images (larger).
height_ratios = []
for _ in all_samples:
    height_ratios.append(1.0)  # title row
    height_ratios.append(4.0)  # images row

gs = gridspec.GridSpec(
    rows, num_images, figure=fig,
    height_ratios=height_ratios
)

for row_idx, (images, title) in enumerate(all_samples):
    # If images has shape (N, 1, H, W), squeeze the channel dimension
    if images.dim() == 4 and images.shape[1] == 1:
        images = images.squeeze(1)

    # ---- Title row ----
    ax_title = fig.add_subplot(gs[2 * row_idx, :])  # entire row
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        title,
        ha="center", va="center",
        fontsize=18
    )

    # ---- Images row ----
    for col_idx in range(num_images):
        ax_img = fig.add_subplot(gs[2 * row_idx + 1, col_idx])
        ax_img.imshow(images[col_idx], cmap="gray")
        ax_img.axis("off")

# 4) Adjust spacing
plt.tight_layout()
plt.show()