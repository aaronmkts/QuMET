import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset

from ...utils import add_dataset_info

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)


@add_dataset_info(
    name="2d_grid_gaussian_b",
    dataset_source="manual",
    available_splits=("train", "validation"),
)
class TwoDGridGaussianDatasetB(Dataset):
    def __init__(self, split="train", n_qubits=16) -> None:

        self.n_qubits = n_qubits
        self.num_dim = 2
        self.num_discrete_values = int(2 ** (n_qubits / self.num_dim))
        self.coords = np.linspace(-3, 3, self.num_discrete_values)
        self.size = 2560 * 10
        self.num_gauss = 9

        if split == "train":
            self.data, _ = self._generate_samples()
        elif split == "validation":
            _, prob_data = self._generate_samples()
            self.data = np.array([prob_data] * self.size)
        else:
            raise RuntimeError(
                f"split must be `train` or `validation`, but got {split}"
            )

    def _generate_samples(self):

        self.set_length = int(pow(self.num_gauss, 1 / 2))

        positions = np.linspace(-1.9, 1.9, int(self.set_length))
        means = np.array(
            [np.array([i, j]) for i, j in itertools.product(positions, positions)]
        )

        sigma = 0.15
        covs = [np.array([[sigma**2, 0], [0, sigma**2]]) for i in range(self.num_gauss)]

        rv = [
            multivariate_normal(mean=mean, cov=cov) for (mean, cov) in zip(means, covs)
        ]

        grid_elements = np.transpose(
            [
                np.tile(self.coords, len(self.coords)),
                np.repeat(self.coords, len(self.coords)),
            ]
        )
        num_samples = len(grid_elements)

        samples = np.sum([dist.pdf(grid_elements) for dist in rv], axis=0)
        prob_data = samples / np.sum(samples)

        index_list = list(range(num_samples))
        sampled_integers = np.random.choice(index_list, size=self.size, p=prob_data)
        grid_bitstrings = np.array(list(map(self._int_to_binary, sampled_integers)))

        return grid_bitstrings, prob_data

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

    def _int_to_binary(self, integer):

        resolution = self.n_qubits
        integer = torch.tensor([integer])
        mask = 2 ** torch.arange(resolution - 1, -1, -1)
        binary = integer.bitwise_and(mask).ne(0).float()

        return binary

    def __len__(self):
        return self.size

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def __getitem__(self, index):

        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)

        return data_i
