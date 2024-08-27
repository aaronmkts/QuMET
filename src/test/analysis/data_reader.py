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
# Prepare temp dirs for storing event files
tmpdirs = {}

def main():
   
    log_dir = '../qumet_output/pqwgan_qc/mnist_0'

    

    df = SummaryReader(log_dir, pivot=True, extra_columns={'dir_name'})
    df = df.scalars
    
    vaegan = df[df['dir_name'] == 'final_result/VAEGAN_gaussian/2000_samples/software/tensorboard/lightning_logs/version_0']
    vaegan_remap = df[df['dir_name'] == 'final_result/VAEGAN_gaussian(remap)/2000_samples/software/tensorboard/lightning_logs/version_1']
    
    plt.plot(vaegan_remap['step'], vaegan_remap['encoder/prior_loss'])
    plt.plot(vaegan['step'], vaegan['encoder/prior_loss'])
  
    # Set custom font size for x and y ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
  
    plt.xlabel('Iterations' , fontsize=16)
    plt.ylabel('Prior Loss', fontsize=16)
    plt.legend(['Weight Remapping', 'Baseline'], fontsize=16)

    plt.show()
     
     
    #%%%%%%%%%%%%%%%%%%%
    '''
    gauss = df[df['dir_name'] == 'final_result/gaussian_prior(no-remap)/2000_samples/software/tensorboard/lightning_logs/version_0']
    uniform = df[df['dir_name'] == 'final_result/uniform_prior(no-remap)/2000_samples/software/tensorboard/lightning_logs/version_0']
    vaegan = df[df['dir_name'] == 'final_result/VAEGAN_gaussian/2000_samples/software/tensorboard/lightning_logs/version_0']
    apqgan = df[df['dir_name'] == 'final_result/VAEGAN_lthlayer/2000_samples/software/tensorboard/lightning_logs/version_0']
    plt.plot(gauss['step'], gauss['wasserstein_distance'])
    plt.plot(uniform['step'], uniform['wasserstein_distance'])
    plt.plot(vaegan['step'], vaegan['wasserstein_distance'])
    plt.plot(apqgan['step'], apqgan['wasserstein_distance'])
    
    # Set custom font size for x and y ticks
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
  
    plt.xlabel('Iterations' , fontsize=16)
    plt.ylabel('Wasserstein Distance', fontsize=16)
    plt.legend(['Gaussian', 'Uniform', 'APQGAN (Gaussian)', 'APQGAN (lth layer)'], fontsize=16)

    plt.show()


    ''' 

    
    
    log_dir = '../qumet_output/pqwgan_qc/mnist_0/final_result/VAEGAN_gaussian/2000_samples/software/tensorboard/lightning_logs/version_0'
    df = SummaryReader(log_dir, pivot=True).images
    df.columns

    #For images
    image = df.loc[9, 'images/sample']

 
    plt.imshow(image.transpose(0, 1, 2))  # Permute to (H, W, C)
    plt.axis('off')
    plt.show()
    
    
if __name__ == "__main__":
    main()

