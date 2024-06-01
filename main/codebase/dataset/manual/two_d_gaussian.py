import numpy as np
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from ..utils import add_dataset_info
import matplotlib.pyplot as plt
from matplotlib import cm
import torch 
# Set the random seed for reproducibility



@add_dataset_info(
    name="2d_gaussian",
    dataset_source="manual",
    available_splits=("train", "validation"),
    discrete_generation=True,
    continuous_generation= True
)
class TwoDGaussianDataset(Dataset):
    def __init__(self, split="train", normaliser = None, discretisation = None, n_qubits=6) -> None:
       
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
        self.n_samples = 2560 * 5
        self.discretisation = discretisation(n_qubits, n_dim=2) if discretisation else None


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
        """Generate 2D Gaussian"""
        # Parameters

        std = 0.05  # Standard deviation of the Gaussians
        mean = np.array([0,0])

        # Covariance matrix for each Gaussian (assuming isotropic Gaussians)
        cov = np.diag([std**2, std**2])
        
        # Generate samples from each Gaussian distribution
        samples = [np.random.multivariate_normal(mean, cov, self.n_samples)]

         # Combine all samples into a single array for easier plotting
        all_samples = np.vstack(samples)
   
        data = self.normaliser.fit_transform(all_samples) if self.normaliser else all_samples

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
        distribution /= np.sum(distribution)
        distribution = np.array(distribution).reshape((num_discrete_values ** 2))
   
        return train_dataset, distribution

    def __len__(self):
        return len(self.data)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def __getitem__(self, index):

        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)
    
        return data_i
