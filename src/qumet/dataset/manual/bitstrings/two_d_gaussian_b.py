import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.utils.data import Dataset
from ...utils import add_dataset_info
import matplotlib.pyplot as plt
from matplotlib import cm

# Set the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

@add_dataset_info(
    name="2d_gaussian_b",
    dataset_source="manual",
    available_splits=("train", "validation"),
    bitstring_generation=True,
)
class TwoDGaussianDatasetB(Dataset):
    def __init__(self, split="train", n_qubits=16) -> None:

        self.n_qubits = n_qubits
        self.num_dim = 2
        self.num_discrete_values = int(2 ** (n_qubits / self.num_dim))
        self.coords = np.linspace(-2, 2, self.num_discrete_values)
        self.size = 2 * 2560

        if split == "train":
            self.data, self.coords = self._generate_samples()
        elif split == "validation":
            _, prob_data = self._generate_samples()
            self.data = np.array([prob_data] * self.size)
        else:
            raise RuntimeError(
                f"split must be `train` or `validation`, but got {split}"
            )

        
    def _generate_samples(self):

        rv = multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 1]], seed=SEED)
        grid_elements = np.transpose(
            [
                np.tile(self.coords, len(self.coords)),
                np.repeat(self.coords, len(self.coords)),
            ]
        )
        num_samples = len(grid_elements)
        
        samples = rv.pdf(grid_elements)
        prob_data = samples / np.sum(samples)

        return prob_data, grid_elements

    def _visualise(self, samples):

        mesh_x, mesh_y = np.meshgrid(self.coords, self.coords)
        grid_shape = (self.num_discrete_values, self.num_discrete_values)
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(samples, grid_shape)
        surf = ax.plot_surface(
            mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def __len__(self):
        return self.size

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def __getitem__(self, index):

        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)
        coords_i = torch.tensor(self.coords[index, ...], dtype=torch.float32)

        return data_i, coords_i
