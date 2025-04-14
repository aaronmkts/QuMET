import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import sys
import os
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "src"))
from qumet.dataset import get_dataset, get_dataset_info
from qumet.models import get_model
from qumet.tools.checkpoint_load import load_model
import io
from PIL import Image
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio 
from qumet.plt_wrapper.metrics import NDB_JSD_Metric
#METRIC CALCULATION

#Cosine Similarity
def calculate_cos(v1, v2):
    v1 = v1.detach().cpu().numpy().reshape(-1, 784)
    v2 = v2.detach().cpu().numpy().reshape(-1, 784)
    num = np.dot(v1, np.array(v2).T) 
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1) 
    res = num / denom
    res[np.isneginf(res)] = 0
    res = 0.5 + 0.5 * res
    cos_mean = np.mean(res)
    return cos_mean

# FID
def calculate_fid(act1, act2):
        """
        Compute FID given two sets of activations or flattened images.
        By default, this is set up for 28x28 images => 784-dim. 
        Adjust as needed if your images have different shape.
        """
        # Move to CPU and flatten
        act1 = act1.detach().cpu().numpy().reshape([-1, 784])
        act2 = act2.detach().cpu().numpy().reshape([-1, 784])

        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)

        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Frechet Distance
        fid_value = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid_value

# PSNR
def calculate_psnr(real_imgs, fake_imgs):
    real = fake_imgs.detach().cpu().numpy().reshape(-1, 28, 28)
    fake = real_imgs.detach().cpu().numpy().reshape(-1, 28, 28)

    psnr_list = []
    for i in range(len(real)):
        psnr_val = peak_signal_noise_ratio(real[i], fake[i])
        psnr_list.append(psnr_val)

    psnr_mean = np.mean(psnr_list)
    return psnr_mean

# SSIM
def calculate_ssim(real_imgs, fake_imgs):
        real = real_imgs.detach().cpu().numpy().reshape(-1, 28, 28)
        fake = fake_imgs.detach().cpu().numpy().reshape(-1, 28, 28)

        ssim_values = []
        for i in range(len(real)):
            ssim_val = structural_similarity(
                real[i], 
                fake[i],
                data_range=1.0  
            )
            ssim_values.append(ssim_val)

        ssim_mean = np.mean(ssim_values)
        return ssim_mean
    

def calculate_ndb_jsd(real_imgs, fake_imgs):
    real = real_imgs.detach().cpu().reshape(-1, 28, 28)
    fake = fake_imgs.detach().cpu().reshape(-1, 28, 28)

    number_of_bins = 50
    significance_level = 0.05
    z_threshold = None
    whitening = None
    max_dims =None
    # Initialize the NDB_JSD_Metric 
    ndb_jsd_metric = NDB_JSD_Metric(
            number_of_bins=number_of_bins,
            significance_level=significance_level,
            z_threshold=z_threshold,
            whitening=whitening,
            max_dims=max_dims
        )
    
    ndb_jsd_metric.update(real, data_type='training')
    ndb_jsd_metric.update(fake, data_type='generated')

    metrics = ndb_jsd_metric.compute()
    ndb_value = metrics['NDB'] / number_of_bins
    jsd_value = metrics['JS']

    return ndb_value, jsd_value

def evaluate_metrics_for_checkpoint(checkpoint_path, real_imgs, n_samples=2600):

    # -------------------------------
    # 1. Load the Pre-trained Model
    # -------------------------------
    APQGAN = get_model("apqgan", "image_generation", get_dataset_info("mnist"))
    model = load_model(checkpoint_path, "pl", APQGAN)
    model.eval()
    
    # -------------------------------
    # 2. Extract Latent Vectors from the Real Images
    # -------------------------------
    with torch.no_grad():
        _, _, z = model.vae_forward(real_imgs)
        z_samples = z  

    z_samples_np = z_samples.cpu().numpy()
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=9).fit(z_samples_np)
    disp_prior, _ = gmm.sample(n_samples)
    disp_prior = torch.tensor(disp_prior, dtype=torch.float32)

    with torch.no_grad():
        fake_imgs = model(disp_prior)
    
    cos_mean   = calculate_cos(real_imgs, fake_imgs)
    fid_value  = calculate_fid(real_imgs, fake_imgs)
    psnr_mean  = calculate_psnr(real_imgs, fake_imgs)
    ssim_mean  = calculate_ssim(real_imgs, fake_imgs)
    ndb_value, jsd_value = calculate_ndb_jsd(real_imgs, fake_imgs)
    
    metrics = {
        "cosine": cos_mean,
        "fid": fid_value,
        "psnr": psnr_mean,
        "ssim": ssim_mean,
        "ndb": ndb_value,
        "jsd": jsd_value
    }
    
    return metrics


def evaluate_checkpoints(checkpoint_paths, real_imgs, n_samples=2600):

    results = []
    for cp in checkpoint_paths:
        metrics = evaluate_metrics_for_checkpoint(cp, real_imgs, n_samples)
        results.append(metrics)
        print("Results for checkpoint {}:".format(cp))
        for metric, value in metrics.items():
            print("  {}: {:.4f}".format(metric, value))
        print("-" * 40)
    
    # Organize results by metric name
    aggregated = {}
    for metric in results[0].keys():
        metric_values = [res[metric] for res in results]
        aggregated[metric] = {
            "mean": np.mean(metric_values),
            "std": np.std(metric_values)
        }
    
    print("Aggregate Results across checkpoints:")
    for metric, stats in aggregated.items():
        print("  {} - Mean: {:.4f}, Std: {:.4f}".format(metric, stats["mean"], stats["std"]))
    
    return aggregated

# ====== Main Execution ======
if __name__ == "__main__":
  
    dataset = get_dataset('fashion_mnist', 'train', 'min-max', False, 2600, 8)
    mnist = dataset.data
    mnist = mnist.to(torch.float32) / 255.0  
    mnist = mnist.unsqueeze(1)
    real_images = mnist.cpu().numpy().squeeze(1)

    checkpoint_paths = [
        "../artifacts/model-vefs28fk:v1/model.ckpt",
        "../artifacts/model-rlx9or4d:v1/model.ckpt",
        "../artifacts/model-j9gn11m7:v1/model.ckpt",
    ]
    
    # -------------------------------
    # 3. Evaluate Metrics Across Checkpoints
    # -------------------------------
    aggregated_results = evaluate_checkpoints(checkpoint_paths, mnist, n_samples=2600)
    print(aggregated_results)
    print("fmnist017real")