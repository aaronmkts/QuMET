from math import sqrt

import numpy as np
import torch
from torch.utils.data import Dataset
from qumet.dataset.utils import add_dataset_info

@add_dataset_info(
    name="bars_and_stripes",
    dataset_source="manual",
    available_splits=("train", "validation"),
    probs_generation=True,
)
class BarsAndStripesDataset(Dataset):
    def __init__(self, split="train", normaliser = None, discretisation = None, n_qubits=9) -> None:
        super().__init__()
        
        self.n_samples = int(sqrt(n_qubits))
        self.n_qubits = n_qubits

        if split in ['train', "validation"]:
            self.data = self._generate_samples()

    def _generate_samples(self):
        n = self.n_samples
        bitstrings = [list(np.binary_repr(I, n))[::-1] for I in range(2 ** n)]
        bitstrings = np.array(bitstrings, dtype=int)

        stripes = bitstrings.copy()
        stripes = np.repeat(stripes, n, 0)
        stripes = stripes.reshape(2 ** n, n * n)

        bars = bitstrings.copy()
        bars = bars.reshape(2 ** n * n, 1)
        bars = np.repeat(bars, n, 1)
        bars = bars.reshape(2 ** n, n * n)

        data = np.vstack((stripes[0: stripes.shape[0] - 1], bars[1: bars.shape[0]]))

        bitstrings = []
        nums = []
        for d in data:
            bitstrings += ["".join(str(int(i)) for i in d)]
            nums += [int(bitstrings[-1], 2)]


        target_probs = np.zeros(2 ** self.n_qubits)
        target_probs[nums] = 1 / len(data)
        target_probs = torch.tensor(target_probs , dtype=torch.float64).unsqueeze(0)
        return target_probs

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        return self.data[index, ...]