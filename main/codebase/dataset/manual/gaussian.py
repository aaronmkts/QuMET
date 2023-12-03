import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from ..utils import add_dataset_info


@add_dataset_info(
    name="gaussian",
    dataset_source="manual",
    available_splits=("train", "test"),
    generation = True,
)
class GaussianDataset(Dataset):
    def __init__(self,  mean, probabilities, std_devs, split = "train", num_gaussians: int = 5, num_samples:int = 1000) -> None:
        
        # Check if the input parameters are valid
        if num_gaussians != len(mean) or num_gaussians != len(probabilities):
            raise ValueError(
                "Number of standard deviations and probabilities should match the number of distributions."
            )

        # Check if probabilities add up to 1
        if np.sum(probabilities) != 1.0:
            raise ValueError("Probabilities should add up to 1.")

        self.num_gaussians = num_gaussians
        self.mean = mean
        self.std_devs = std_devs
        self.probabilities = probabilities
        self.num_samples = num_samples

        self.samples = self._generate_samples()
        self.train, self.test = train_test_split(self.samples, train_size = 0.7, shuffle = True)

        if split == "train":
            self.data = np.array(self.train).reshape((-1,1))
        elif split == "test":
            self.data = np.array(self.test).reshape((-1,1))
        else:
            raise RuntimeError(
                f"split must be `train` or `test`, but got {split}"
            )
        
    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def _generate_samples(self):

        samples = []

        for _ in range(self.num_samples):
        
            chosen_distribution = np.random.choice(np.arange(self.num_gaussians), p=self.probabilities)

            # Generate a random sample from the chosen distribution
            sample = np.random.normal(self.mean[chosen_distribution], self.std_devs)
            samples.append(sample)

        return samples


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        
        data_i = torch.tensor(self.data[index, ...], dtype=torch.float32)

        return data_i
