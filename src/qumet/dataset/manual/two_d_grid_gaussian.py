import numpy as np
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from ..utils import add_dataset_info
import matplotlib.pyplot as plt
from matplotlib import cm
import torch 
import itertools
# Set the random seed for reproducibility



@add_dataset_info(
    name="2d_grid_gaussian",
    dataset_source="manual",
    available_splits=("train", "validation"),
    bitsring_generation=True,
    continuous_generation=True
)
class TwoDGridGaussianDataset(Dataset):
    def __init__(self, split="train", normaliser = None, discretisation = None, n_qubits=6) -> None:
        super().__init__()
        """
        Initialize the TwoDGaussianDataset.

        Args:
            split (str): Dataset split type ('train' or 'validation').
            normaliser: Normalization object with fit_transform and reverse_lookup methods.
            discretisation: Discretization function.
            n_qubits (int): Number of qubits for discretization.
        """

        self.n_qubits = n_qubits
        self.normaliser = normaliser
        self.reverse_lookup = normaliser.reverse_lookup if normaliser else None
        self.n_dim = 2
        self.n_samples = 25600
        self.discretisation = discretisation(n_qubits, n_dim=2) if discretisation else None
        self.n_gauss = 9

        if split == "train":
            self.data, _ = self._generate_samples()
        elif split == "validation":
            _, prob_data = self._generate_samples()
            self.data = np.array([prob_data] * self.n_samples)
        else:
            raise RuntimeError(
                f"split must be `train` or `validation`, but got {split}"
            )

    def _generate_samples(self):

        # Parameters
        std = 0.3
        cov = np.diag([std**2, std**2])

        step_size = int(pow(self.n_gauss, 1 / 2))
        positions = np.linspace(-1.9, 1.9, int(step_size))
        means = np.array(
            [np.array([i, j]) for i, j in itertools.product(positions, positions)]
        )
        n_samples_per_gauss = self.n_samples // self.n_gauss
        extra_samples = self.n_samples % self.n_gauss

        # Create distribution objects for each Gaussian
        rv = [multivariate_normal(mean=mean, cov=cov) for mean in means]
        # Generate samples
        samples = np.zeros((self.n_samples, 2))
        component_indices = np.hstack([np.full(n_samples_per_gauss + (1 if i < extra_samples else 0), i) for i in range(self.n_gauss)])
        np.random.shuffle(component_indices)
        
        for i, component_index in enumerate(component_indices):
            samples[i] = rv[component_index].rvs()

        data = self.normaliser.fit_transform(samples) if self.normaliser else samples

        if self.discretisation:
            data, distribution = self._discretise_samples(data)
            return data, distribution
        return data 

    def _discretise_samples(self, data):
        
        num_discrete_values = int(2 ** (self.n_qubits / self.n_dim)) #discretisation per dimension
        nns = tuple(num_discrete_values for _ in range(self.n_dim)) #siply (n,n) for 2 d and (n,n,n) for 3d data
        nns_nq = nns + tuple((self.n_qubits,)) #(n,n, n_qubits) 8 by 8 grid with qubits appended

        inverse_bins = np.zeros(nns_nq) #empty matrix with shape ((n,n, n_qubits))
        for key, value in self.discretisation.items():
            id_n = value[0]
            inverse_bins[id_n] = np.array([int(bit) for bit in key])

        coordinates = np.floor(data * num_discrete_values).astype(int)
        train_dataset = np.array([inverse_bins[tuple(coord)] for coord in coordinates])

        distribution = np.zeros(nns)
        for xy in coordinates:
            indices = tuple(xy[ii] for ii in range(self.n_dim))
            distribution[indices] += 1
        # Add a small value to empty elements

        distribution /= np.sum(distribution)
        distribution = np.array(distribution).reshape((num_discrete_values ** 2))
   
        return train_dataset, distribution
    
    def __len__(self):
        return self.n_samples

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def __getitem__(self, index):

        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)

        return data_i
