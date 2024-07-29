"""
Pytorch & Pennylane Hybrid Qgan model
Code Adapted from https://github.com/SilverEngineered/MosaiQ/blob/main/mosaiq.py
"""

# Library imports
import math
import random
import numpy as np
from torch import Tensor
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import partial
import matplotlib.pyplot as plt
from typing import Optional as _Optional
# Pytorch & Pennylane imports
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer as TorchConnector
from pennylane import broadcast
from pennylane.wires import Wires



logger = getLogger(__name__)
pi = math.pi

# fmt:0ff
pca_dims = 40

config = {
    "discriminator": {"input_size": pca_dims},
    "generator": {
        "device": "default.qubit",
        "n_qubits": 5,
        "n_a_qubits": 0,
        "shots": 10000,
        "depth": 6,
        "q_delta": 1,
        "diff_method": "adjoint",
        "n_generators": 8,
        "q_delta": 1

    },
}
ordering = []
for i in range(8):
    k = 4* i
    l = [i, 39-k, 38-k, 37-k, 36-k]
    ordering.append(l)

image_size = 5
#DISCRIMINATOR

class Discriminator(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        name = "discriminator"
        self.input_size = config[name]["input_size"]

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    


#GENERATOR
name = "generator"
generator_config = config[name]

device = generator_config["device"]
n_qubits = generator_config["n_qubits"]
n_a_qubits = generator_config["n_a_qubits"]
depth = generator_config["depth"]


dev = qml.device("default.qubit", wires=n_qubits)

######################################################################
# Next, we define the quantum circuit and measurement process described above.
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    weights = weights.reshape(depth, n_qubits)
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
        qml.RX(noise[i], wires=i)
    for i in range(depth):
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])
    return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

class MosaiQGenerator(nn.Module):
    def __init__(self, config, task):
        
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        name = "generator"
        n_generators = config[name]["n_generators"]
        q_delta = config[name]["q_delta"]
        self.n_qubits = config[name]["n_qubits"]
        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(depth, n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )

        self.n_generators = n_generators

    def forward(self, x, batch_size):
        images = []
        patch_size = image_size
        images = torch.Tensor(x.size(0), 0)
        for params in self.q_params:
            patches = torch.Tensor(0, patch_size)
            for elem in x:
                f = quantum_circuit(elem, params)
                f = torch.stack(f)
                q_out = f.float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            flattened_order =  [j for sub in ordering for j in sub]
            patches = torch.flatten(patches)
            patches = patches[flattened_order] # Rearrange order of pca components
            patches = patches.reshape(batch_size, patch_size)
            images = torch.cat((images, patches), 1)
        return images
    


class MosaiQ(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks

        self.discriminator = Discriminator(config, task)
        self.generator = MosaiQGenerator(config, task)

    def forward(self, input: Tensor):
        return self.generator(input)


# ---------------------------------------
# MosaiQ
# ---------------------------------------


def _mosaiq(config, task: str) -> MosaiQ:

    model = MosaiQ(config, task)
    return model


def get_mosaiq(info: Dict) -> MosaiQ:

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into MosaiQ")
    return _mosaiq(config=config, task=task)