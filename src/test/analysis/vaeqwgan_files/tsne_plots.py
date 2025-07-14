import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import seaborn as sns

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"..", "src"
     )
    )
from qumet.dataset import QuMETDataModule
from qumet.models import get_model, get_model_info
from qumet.dataset import get_dataset, get_dataset_info
import torch.optim as optim
from qumet.tools.checkpoint_load import *
import wandb
import re
from sklearn.mixture import GaussianMixture




def main():
    import torch
    from torchvision import datasets, transforms
    from sklearn.manifold import TSNE
    
    # Load MNIST data using torchvision
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
  
    # Extract data and labels
    X = train_dataset.data
    y = train_dataset.targets

    # Flatten the images from 28x28 to 784
    X = X.view(-1, 28*28).numpy()
    y = y.numpy()

    # ---- NEW PART: Subselect images that correspond to labels 1, 3, and 5 ----
    mask = np.isin(y, [0, 1, 7, 8])
    X = X[mask]
    y = y[mask]

    # Subselect 2000 random samples to save time (optional; can adjust as needed)
    np.random.seed(42)
    idx = np.random.choice(len(X), 3000, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    # Perform TSNE# Remap labels to a contiguous range
    unique_labels = np.unique(y_sub)
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    y_mapped = np.array([label_mapping[label] for label in y_sub])

    ''' 
    APQGAN = get_model("vaeqwgan", "image_generation", get_dataset_info("mnist"))
    checkpoint_path_vae = "artifacts/model-6x4m1nb1:v1/model.ckpt"
    model_vaeqwgan = load_model(checkpoint_path_vae, "pl", APQGAN)
    model_vaeqwgan.eval()
    
    latent_z = []
    model_vaeqwgan.eval()
    with torch.no_grad():
        X_new = torch.tensor(X_sub, dtype=torch.float32) / 255.0
        X_new = X_new.view(-1,1,28,28)  # Add channel dimension
        
        mu, log_var, z = model_vaeqwgan.vae_forward(X_new)
        latent_z.append(z)
    z_samples = torch.cat(latent_z, dim=0).cpu().numpy()

    n_samples = 3000  
    gmm = GaussianMixture(n_components=7, covariance_type='tied', random_state=9).fit(z_samples)
    disp_prior, _ = gmm.sample(n_samples)
    disp_prior = torch.tensor(disp_prior, dtype=torch.float32)

    model_vaeqwgan.eval()
    with torch.no_grad():
        fake_images = model_vaeqwgan(disp_prior)
    X_fake = fake_images.view(-1, 28* 28)

    '''
    ## Perform TSNE
    tsne = TSNE(n_components=2, random_state=32)
    X_tsne = tsne.fit_transform(X_sub)

    # Plot the results
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_mapped,
        cmap=plt.cm.get_cmap("tab10", len(unique_labels)),
        alpha=0.7
    )
    ''' 
    X_tsne2 = tsne.fit_transform(X_fake)
    
    plt.scatter(
        X_tsne2[:, 0],
        X_tsne2[:, 1],
        c='black',
        label='Generated Samples',
        alpha=0.8
    )
    '''

    cbar = plt.colorbar(sc, ticks=range(len(unique_labels)))
    cbar.ax.set_yticklabels(unique_labels)               # show true digit labels
    cbar.set_label("Digit label", rotation=270, labelpad=15)



    plt.axis('off')
    plt.show()
if __name__ == "__main__":
    main()
