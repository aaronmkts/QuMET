import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.utils.data import Dataset
from ..utils import add_dataset_info
import matplotlib.pyplot as plt
from matplotlib import cm
# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)

@add_dataset_info(
    name="2dgaussian",
    dataset_source="manual",
    available_splits=("train", "test"),
    generation = True,
)
class TwoDGaussianDataset(Dataset):
    def __init__(self,  split = "train") -> None:

        self.num_discrete_values =  8  #(2 ** n_qubits)
        self.coords = np.linspace(-2, 2, self.num_discrete_values)
        self.samples = self._generate_samples()
        

        if split == "train":
            self.data = np.array(self.samples).reshape((-1,1))
        elif split == "test":
            self.data = np.array(self.samples).reshape((-1,1))
        else:
            raise RuntimeError(
                f"split must be `train` or `test`, but got {split}"
            )
        
    def _generate_samples(self):

        samples = []

        #automate mean placement and suitable sample size (2 ** n_qubits)
        rv = multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 1]], seed=seed)
        grid_elements = np.transpose([np.tile(self.coords, len(self.coords)), np.repeat(self.coords, len(self.coords))])
        prob_data = rv.pdf(grid_elements)
        prob_data = prob_data / np.sum(prob_data)
        samples = prob_data
        return samples
        
    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass
    
    def visualise(self):

        mesh_x, mesh_y = np.meshgrid(self.coords, self.coords)
        grid_shape = (self.num_discrete_values, self.num_discrete_values)
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(self.samples, grid_shape)
        surf = ax.plot_surface(mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)

        return data_i
