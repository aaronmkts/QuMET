import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,".."
     )
    )
import torch 
import torch.nn as nn
import tempfile
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import torchvision
import pandas as pd
def main():
    
        # Define the metric to plot by assigning the column name
   

        # Define the log directories
    log_dir_gauss = '../qumet_output/icassp2024/final_results/fashionmnist/gaussian_prior'
    log_dir_qvaegan = '../qumet_output/icassp2024/final_results/fashionmnist/qvaegan'
    log_dir_uniform = '../qumet_output/icassp2024/final_results/fashionmnist/uniform_prior'

    # Function to process model data and return the image
    def process_model_data(log_dir):
        df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).images
        seed19 = df[df['dir_name'] == 'seed_23/software/tensorboard/lightning_logs/version_0']
        image = df.loc[9, 'images/sample']
        return image

    def process_model_data2(log_dir):
        df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).images
        seed19 = df[df['dir_name'] == 'seed_19/software/tensorboard/lightning_logs/version_0']
        image = df.loc[9, 'images/disp_prior']
        return image

    def process_model_data3(log_dir):
        df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'}).images
        seed19 = df[df['dir_name'] == 'seed_23/software/tensorboard/lightning_logs/version_0']
        image = df.loc[9, 'images/real']
        return image

    # Load images from Gauss, VAEGAN, and Uniform prior models
    image_gauss = process_model_data(log_dir_gauss)
    image_qvaegan = process_model_data(log_dir_qvaegan)
    image_uniform = process_model_data(log_dir_uniform)
    image_disp = process_model_data2(log_dir_qvaegan)
    image_real = process_model_data3(log_dir_qvaegan)

    # Create a figure with minimal space between the subplots
    fig, axs = plt.subplots(5, 1, figsize=(6, 6))

    # Plotting the images
    axs[0].imshow(image_real.transpose(0, 1, 2))
    axs[0].axis('off')
    axs[0].set_title('Real images', fontsize=30)  # Set title font size to 30

    axs[1].imshow(image_qvaegan.transpose(0, 1, 2))
    axs[1].axis('off')
    axs[1].set_title(r'$\text{VAE-QWGAN} \ q_{\omega}(z \vert{} x)$', fontsize=30)  # Set title font size to 30

    axs[2].imshow(image_disp.transpose(0, 1, 2))
    axs[2].axis('off')
    axs[2].set_title(r'$\text{GMM}(\mu,\Sigma)$', fontsize=30)  # Set title font size to 30

    axs[3].imshow(image_uniform.transpose(0, 1, 2))
    axs[3].axis('off')
    axs[3].set_title(r'Uniform $U_{[0,1)}$', fontsize=30)  # Set title font size to 30

    axs[4].imshow(image_gauss.transpose(0, 1, 2))
    axs[4].axis('off')
    axs[4].set_title(r'Gaussian $\mathcal{N}(0, \mathbb{I})$', fontsize=30)  # Set title font size to 30

    # Adjust spacing between subplots and remove extra space around the figure
    plt.subplots_adjust(wspace=0, hspace=0.05, top=0.95, bottom=0.05, left=0.01, right=0.99)

    # Show the plot with trimmed whitespace
    plt.show()

        
    
if __name__ == "__main__":
    main()

