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
from itertools import product
from codebase.dataset import QuMETDataModule
from codebase.models import get_model, get_model_info
from codebase.dataset import get_dataset

from codebase.plt_wrapper import get_model_wrapper
import pytorch_lightning as L
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import multivariate_normal
import io
import torchvision
import tensorflow as tf
import torch
from codebase.dataset.manual.transforms.utils import MinMaxNormalizer, PITNormalizer
import pennylane as qml
from pennylane.transforms import insert
from functools import partial


def main():
    class MinMaxNormalizer:
        def __init__(self, reverse_lookup = None, epsilon = 0):
            self.reverse_lookup = reverse_lookup
            self.epsilon = epsilon

        def fit_transform(self, data: np.ndarray) -> np.ndarray:
            self.min = data.min()
            self.max = data.max() - data.min()
            data = (data - self.min) / self.max
            self.reverse_lookup = (self.min, self.max)
            return data / (1 + self.epsilon)

        def transform(self, data: np.ndarray) -> np.ndarray:
            min = data.min()
            max = data.max() - data.min()
            data = (data - min) / max
            return data / (1 + self.epsilon)

        def inverse_transform(self, data: np.ndarray) -> np.ndarray:
            data = data * (1 + self.epsilon)
            self.min, self.max = self.reverse_lookup
            return data * self.max + self.min
    # Parameters
    def center(coord, n):
        return np.array(coord) / n + 0.5 / n
    
    def compute_discretization(n_qubits, n_dim):
        format_string = "{:0" + str(n_qubits) + "b}"
        n = 2 ** (n_qubits // n_dim)
        dict_bins = {}

        for k, coordinates in enumerate(product(range(n), repeat=n_dim)):
            dict_bins.update({
                format_string.format(k): [coordinates, center(coordinates, n)]
            })
        return dict_bins
    
    def discretise_samples(data):

        
        num_discrete_values = int(2 ** (6 / 2)) #discretisation per dimension
        nns = tuple(num_discrete_values for _ in range(2)) #siply (n,n) for 2 d and (n,n,n) for 3d data
        nns_nq = nns + tuple((6,)) #(n,n, n_qubits) 8 by 8 grid with qubits appended

        inverse_bins = np.zeros(nns_nq) #empty matrix with shape ((n,n, n_qubits))
        for key, value in discretisation.items():
            id_n = value[0]
            inverse_bins[id_n] = np.array([int(bit) for bit in key])

        coordinates = np.floor(data * num_discrete_values).astype(int)

        train_dataset = np.array([inverse_bins[tuple(coord)] for coord in coordinates])

        distribution = np.zeros(nns)
        for xy in coordinates:
            indices = tuple(xy[ii] for ii in range(2))
            distribution[indices] += 1
        # Add a small value to empty elements

        distribution /= np.sum(distribution)
        distribution = np.array(distribution).reshape((num_discrete_values ** 2))

        return train_dataset, distribution
    
    discretisation = compute_discretization(6,2)
    normaliser = MinMaxNormalizer(epsilon=1e-8)

    def datasetB():
        num_gauss = 9
        set_length = int(pow(num_gauss, 1 / 2))
        num_discrete_values = int(2 ** (6 / 2))
        coords = np.linspace(-3, 3, num_discrete_values)

        positions = np.linspace(-1.9, 1.9, int(set_length))
        means = np.array(
            [np.array([i, j]) for i, j in itertools.product(positions, positions)]
        )

        sigma = 0.15
        covs = [np.array([[sigma**2, 0], [0, sigma**2]]) for i in range(num_gauss)]

        rv = [
            multivariate_normal(mean=mean, cov=cov) for (mean, cov) in zip(means, covs)
        ]

        grid_elements = np.transpose(
            [
                np.tile(coords, len(coords)),
                np.repeat(coords, len(coords)),
            ]
        )

        num_samples = len(grid_elements)

        samples = np.sum([dist.pdf(grid_elements) for dist in rv], axis=0)
        breakpoint()
        prob_data = samples / np.sum(samples)

        return prob_data

    

    def datasetA():
        n_samples = 50000
        std = 0.05
        cov = np.diag([std**2, std**2])
        n_gauss = 25
        step_size = int(pow(n_gauss, 1 / 2))
        positions = range(-4,5,2)
        means = np.array(
            [np.array([i, j]) for i, j in itertools.product(positions, positions)]
        )

        def linear_search_optimized(arr, target):
            for i, num in enumerate(arr):
                if num == target:
                    return i
            return 'not found'
        
        n_samples_per_gauss = n_samples // n_gauss
        extra_samples = n_samples % n_gauss

        # Create distribution objects for each Gaussian
        rv = [multivariate_normal(mean=mean, cov=cov) for mean in means]

        # Generate sampless
        samples = np.zeros((n_samples, 2))
        component_indices = np.hstack([np.full(n_samples_per_gauss + (1 if i < extra_samples else 0), i) for i in range(n_gauss)])
        np.random.shuffle(component_indices)

        for i, component_index in enumerate(component_indices):
            samples[i] = rv[component_index].rvs()
        
        data = samples

        return data
    
    data = datasetA()
    np.save('MG25', data)
    breakpoint()

    

    
if __name__ == "__main__":
    main()
