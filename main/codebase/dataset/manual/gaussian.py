import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from ..utils import add_dataset_info

# Set the random seed for reproducibility
seed = 42
np.random.seed(seed)

@add_dataset_info(
    name="gaussian",
    dataset_source="manual",
    available_splits=("train", "test"),
    generation = True,
)
class GaussianDataset(Dataset):
    def __init__(self,  split = "train") -> None:
        

        self.mean = [-0.2, 0.2]
        self.probabilities = [1/2,1/2]
        self.std_devs = [0.075,0.075]
        self.num_gaussians = 2
        self.num_samples = 1000
       
       # Check if the input parameters are valid
        if self.num_gaussians != len(self.mean) or self.num_gaussians != len(self.probabilities):
            raise ValueError(
                "Number of standard deviations and probabilities should match the number of distributions."
            )

        # Check if probabilities add up to 1
        if np.sum(self.probabilities) != 1.0:
            raise ValueError("Probabilities should add up to 1.")

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
        for _ in range(self.num_samples):
        
            chosen_distribution = np.random.choice(np.arange(self.num_gaussians), p=self.probabilities)

            # Generate a random sample from the chosen distribution
            sample = np.random.normal(loc = self.mean[chosen_distribution], scale = self.std_devs[chosen_distribution])
            samples.append(sample)

        return samples
        
    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        
        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)

        return data_i
    