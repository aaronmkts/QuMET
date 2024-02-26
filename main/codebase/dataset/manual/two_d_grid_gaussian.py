import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.utils.data import Dataset
import itertools
from ..utils import add_dataset_info
import matplotlib.pyplot as plt
from matplotlib import cm

# Set the random seed for reproducibility

seed = 42
np.random.seed(seed)


@add_dataset_info(
    name="2d_grid_gaussian",
    dataset_source="manual",
    available_splits=("train", "validation"),
    generation=True,
)
class TwoDGridGaussianDataset(Dataset):
    def __init__(self, split="train", binary=False, n_qubits=16) -> None:

        self.binary = binary
        self.n_qubits = n_qubits
        self.num_dim = 2
        self.num_discrete_values = int(2 ** (n_qubits / self.num_dim))
        self.coords = np.linspace(-3, 3, self.num_discrete_values)
        self.num_gauss = 16
        self.samples, self.grid_elements = self._generate_samples()

        if split == "train":
            self.data = np.array(self.samples).reshape((-1, 1))

        elif split == "validation":
            self.data = np.array(self.samples).reshape((-1, 1))
        else:
            raise RuntimeError(
                f"split must be `train` or `validation`, but got {split}"
            )

    def _generate_samples(self):

        self.set_length = int(pow(self.num_gauss, 1 / 2))

        positions = np.linspace(-2, 2, int(self.set_length))
        means = np.array(
            [np.array([i, j]) for i, j in itertools.product(positions, positions)]
        )

        sigma = 0.1
        covs = [np.array([[sigma, 0], [0, sigma]]) for i in range(self.num_gauss)]

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

        if self.binary == True:

            index_list = list(range(num_samples))
            sampled_integers = np.random.choice(
                index_list, size=num_samples, p=prob_data
            )
            grid_elements = np.array(list(map(self._int_to_binary, sampled_integers)))

        return prob_data, grid_elements

    def _visualise(self):

        mesh_x, mesh_y = np.meshgrid(self.coords, self.coords)
        grid_shape = (self.num_discrete_values, self.num_discrete_values)
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(self.samples, grid_shape)
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
        return len(self.samples)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def __getitem__(self, index):

        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label_i = torch.tensor(self.grid_elements[index, ...], dtype=torch.float32)

        return data_i, label_i
