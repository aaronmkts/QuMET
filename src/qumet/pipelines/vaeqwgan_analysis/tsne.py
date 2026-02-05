import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "src"
    )
)
import io

from PIL import Image
from torchvision import transforms

from qumet.dataset import get_dataset, get_dataset_info
from qumet.models import get_model
from qumet.tools.checkpoint_load import load_model

# -------------------------------
# Helper Functions
# -------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def compute_tsne(data, random_state=9):
    """
    Compute the t-SNE embedding for the given data.
    data: torch.Tensor (or convertible to torch.Tensor)
    Returns a (N, 2) numpy array.
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data)
    data_np = data.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=random_state)
    emb = tsne.fit_transform(data_np)
    return emb


def plot_tsne(real_tsne, real_labels, fake_tsne, epoch, unique_labels):
    """
    Create a scatter plot of real and fake images in t-SNE space.
      - real_tsne, fake_tsne: (N, 2) arrays
      - real_labels: (N,) array or None
    This function follows your callback plotting style.
    Returns a tensor image of the plot.
    """
    plt.figure(figsize=(10, 8))
    if real_tsne is not None:
        if real_labels is not None:
            n_labels = len(unique_labels)
            label_mapping = {old: new for new, old in enumerate(unique_labels)}
            real_labels_mapped = np.array(
                [label_mapping[label] for label in real_labels]
            )
            scatter = plt.scatter(
                real_tsne[:, 0],
                real_tsne[:, 1],
                c=real_labels_mapped,
                cmap=plt.cm.get_cmap("tab10", n_labels),
                alpha=0.7,
                label="Real Data",
            )
            # cbar = plt.colorbar(scatter)
            # cbar.set_ticks(range(n_labels))
            # cbar.set_ticklabels(unique_labels)
        else:
            plt.scatter(
                real_tsne[:, 0], real_tsne[:, 1], c="blue", alpha=0.5, label="Real Data"
            )
        if fake_tsne is not None:
            plt.scatter(
                fake_tsne[:, 0],
                fake_tsne[:, 1],
                c="black",
                alpha=0.75,
                marker="X",
                label="Fake Data",
            )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # Save the plot to a buffer and convert to a tensor image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_img = Image.open(buf)
    transform = transforms.ToTensor()
    plot_tensor = transform(plot_img)
    plt.close()
    buf.close()
    return plot_tensor


# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------

dataset = get_dataset("fashion_mnist", "train", "min-max", False, 2600, 8)
mnist = dataset.data
real_labels = dataset.targets

mnist = mnist.to(torch.float32) / 255.0
mnist = mnist.unsqueeze(1)  # Ensure shape is [N, 1, H, W]

# Convert real images to numpy and flatten for t-SNE later.
real_images = mnist.cpu().numpy().squeeze(1)
n_real = real_images.shape[0]
real_images_flat = real_images.reshape(n_real, -1)

if torch.is_tensor(real_labels):
    real_labels = real_labels.cpu().numpy()

# -------------------------------
# 2. Load the Pre-trained Models
# -------------------------------

# APQGAN
PQWGAN_GAUSSIAN = get_model("gan", "image_generation", get_dataset_info("mnist"))
checkpoint_path = "../artifacts/model-shm89nxi:v1/model.ckpt"
model = load_model(checkpoint_path, "pl", PQWGAN_GAUSSIAN)

""" 
latent_z = []
model.eval()
with torch.no_grad():
    mu, log_var, z = model.vae_forward(mnist)
    latent_z.append(z)
z_samples = torch.cat(latent_z, dim=0).cpu().numpy()

n_samples = 2600  
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=9).fit(z_samples)
disp_prior, _ = gmm.sample(n_samples)
disp_prior = torch.tensor(disp_prior, dtype=torch.float32)

model.eval()
with torch.no_grad():
    fake_images = model(disp_prior)
fake_images_np = fake_images.cpu().numpy().squeeze(1)
fake_images_flat = fake_images_np.reshape(fake_images_np.shape[0], -1)
"""

z_gauss = torch.rand(2600, 7)
with torch.no_grad():
    fake_images = model(z_gauss)
fake_images_np = fake_images.cpu().numpy().squeeze(1)
fake_images_flat = fake_images_np.reshape(fake_images_np.shape[0], -1)

# -------------------------------
# 4. Prepare Data for t-SNE
# -------------------------------
real_subset = real_images_flat
real_subset_labels = real_labels
fake_subset = fake_images_flat

# -------------------------------
# 5. Compute t-SNE Embeddings
# -------------------------------
# Convert data to torch.Tensor and compute embeddings.
real_tensor = torch.tensor(real_subset)
fake_tensor = torch.tensor(fake_subset)
real_tsne = compute_tsne(real_tensor, random_state=9)
fake_tsne = compute_tsne(fake_tensor, random_state=9)

# -------------------------------
# 6. Plot t-SNE using the Callback-style Method
# -------------------------------
unique_labels = np.unique(real_subset_labels)
epoch = 0
plot_tensor = plot_tsne(real_tsne, real_subset_labels, fake_tsne, epoch, unique_labels)

# Optionally, display the resulting image.
plt.figure(figsize=(10, 8))
plt.imshow(plot_tensor.permute(1, 2, 0))
plt.axis("off")
plt.show()
