import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(
     os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "..", "..", ".." ,"main"
     )
    )
import torch 
import torch.nn as nn

import itertools
from codebase.models.qgan.qgcd_probs.modelling_qgan_probs import QGCD_Probs_GAN, _qgcd_gan
from codebase.models.qgan.qgcd_probs.configuration_qgan_probs import QGCD_Probs_Config
from codebase.dataset import QuMETDataModule
from codebase.models import get_model, get_model_info
from codebase.dataset import get_dataset_info

from codebase.plt_wrapper import get_model_wrapper
import pytorch_lightning as L
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal
import io
import torchvision
import tensorflow as tf
def main():
    '''
    logdir = "logs/plots/" 
    file_writer = tf.summary.create_file_writer(logdir)

    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def image():
        num_discrete_values = 8
        coords = np.linspace(-2, 2, num_discrete_values)

        rv = multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 1]], seed=42)
        grid_elements = np.transpose([np.tile(coords, len(coords)), np.repeat(coords, len(coords))])
        prob_data = rv.pdf(grid_elements)
        prob_data = prob_data / np.sum(prob_data)
        mesh_x, mesh_y = np.meshgrid(coords, coords)
        grid_shape = (num_discrete_values, num_discrete_values)

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(prob_data, grid_shape)
        surf = ax.plot_surface(mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig
    
    figure = image()
    
    with file_writer.as_default():
        tf.summary.image("Training data", plot_to_image(figure), step=0)
    '''

    def bars_and_stripes(rows, cols):
    
        data = [] 
        
        for h in itertools.product([0,1], repeat=cols):
            pic = np.repeat([h], rows, 0)
            data.append(pic.ravel().tolist())
            
        for h in itertools.product([0,1], repeat=rows):
            pic = np.repeat([h], cols, 1)
            data.append(pic.ravel().tolist())
        
        data = np.unique(np.asarray(data), axis=0)
        
        return data
    
    n , m =  2, 3

    bas = bars_and_stripes(n,m)
    print(bas)
    n_points, n_qubits  =  bas.shape

    print(n_points,n_qubits)
    fig, ax_b = plt.subplots(1, bas.shape[0], figsize=(14,2))   #visualization of bars ans stripes data set

    for i in range(bas.shape[0]):
        ax_b[i].matshow(bas[i].reshape(n, m), vmin=-1, vmax=1)
        
        ax_b[i].set_xticks([])
        ax_b[i].set_yticks([])
        
        ax_b[i].set_xticks([0.5], minor=True)
        ax_b[i].set_yticks([0.5], minor=True)
        
        ax_b[i].set_title(bas[i])
        ax_b[i].grid(which='minor', color='black', linestyle='-', linewidth=0.75)

    plt.show()

if __name__ == "__main__":
    main()
