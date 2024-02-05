import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.utils.data import Dataset
import itertools
from ..utils import add_dataset_info
import matplotlib.pyplot as plt
from matplotlib import cm
import math
# Set the random seed for reproducibility and constants
pi = math.pi
seed = 42
np.random.seed(seed)

@add_dataset_info(
    name="2d_ring_gaussian",
    dataset_source="manual",
    available_splits=("train", "test"),
    generation = True,
)
class TwoDRingGaussianDataset(Dataset):
    def __init__(self,  split = "train") -> None:

        self.num_discrete_values =  256 #(2 ** n_qubits)
        self.coords = np.linspace(-3, 3, self.num_discrete_values)
        self.num_gauss = 8
        self.samples, self.grid_elements = self._generate_samples()
        

        if split == "train":
            self.data = np.array(self.samples).reshape((-1,1))
        elif split == "test":
            self.data = np.array(self.samples).reshape((-1,1))
        else:
            raise RuntimeError(
                f"split must be `train` or `test`, but got {split}"
            )
        
    def _generate_samples(self):
        
        self.set_length = int(pow(self.num_gauss, 1/2)) 
        positions  = np.linspace(-2,2,int(self.set_length))
        means = self._means_ring()
        sigma = 0.1
        covs = [np.array([[sigma,0], [0,sigma]]) for i in range(self.num_gauss)]

        rv = [multivariate_normal(mean=mean, cov=cov) for (mean, cov) in zip(means,covs)]

        grid_elements = np.transpose([np.tile(self.coords, len(self.coords)), np.repeat(self.coords, len(self.coords))])
        prob_data = np.sum([dist.pdf(grid_elements) for dist in rv], axis=0)
        samples = prob_data / np.sum(prob_data)
     
        return samples, grid_elements
    
    def _means_ring(self):
        
        self.radius = 2
        means_list = []

        for i in range(self.num_gauss):
            
            theta = ((2 *  pi ) / self.num_gauss) * i 

            x = self.radius * math.sin(theta)
            y = self.radius * math.cos(theta)
            
            means_list.append((x,y))

        means = np.array([np.array([i, j]) for i, j in means_list])

        return means



    def _visualise(self):

        mesh_x, mesh_y = np.meshgrid(self.coords, self.coords)
        grid_shape = (self.num_discrete_values, self.num_discrete_values)
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(self.samples, grid_shape)
        surf = ax.plot_surface(mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def __len__(self):
        return len(self.samples)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def __getitem__(self, index):

        data_i = torch.tensor(self.data[index, ...], dtype = torch.float32)
        label_i = torch.tensor(self.grid_elements[index, ...], dtype = torch.float32)

        return data_i, label_i