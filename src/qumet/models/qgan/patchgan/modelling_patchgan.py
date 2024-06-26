"""
Pytorch & Pennylane Hybrid Qgan model
Code Adapted from https://pennylane.ai/qml/demos/tutorial_quantum_gans.html
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
config = {
    "discriminator": {"input_size": 8}, #Due to image resize, height/width is 8
    "generator": {
        "device": "lightning.qubit",
        "n_qubits": 5,
        "n_a_qubits": 1,
        "shots": 10000,
        "depth": 6,
        "q_delta": 1,
        "diff_method": "adjoint",
        "n_generators": 4,
        "q_delta": 1

    },
}

#DISCRIMINATOR

# fmt:on
class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config, task):
        super().__init__()
        name = "discriminator"
        self.input_size = config[name]["input_size"]

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(self.input_size * self.input_size, 64),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 16),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
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


dev = qml.device(device, wires=n_qubits)

######################################################################
# Next, we define the quantum circuit and measurement process described above.
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    weights = weights.reshape(depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(depth):
        # Parameterised layer
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))

# For further info on how the non-linear transform is implemented in Pennylane
# https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
def partial_measure(noise, weights):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven

class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, config, task):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """
        name = "generator"
        n_generators = config[name]["n_generators"]
        q_delta = config[name]["q_delta"]
        self.n_qubits = generator_config["n_qubits"]
        super().__init__()

        self.q_weights = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators
      
    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (n_qubits - n_a_qubits)
        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0)
        # Iterate over all sub-generators

        for params in self.q_weights:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size)
            # for b in batch basically
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))
            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)
            
        return images
    


class PatchGAN(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks
      
        self.discriminator = Discriminator(config, task)
        self.generator = PatchQuantumGenerator(config, task)

    def forward(self, input: Tensor):
        return self.generator(input)


# ---------------------------------------
# PatchGAN
# ---------------------------------------


def _patchgan(config, task: str) -> PatchGAN:

    model = PatchGAN(config, task)
    return model


def get_patchgan(info: Dict) -> PatchGAN:

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into PatchQuantumGenerator")
    return _patchgan(config=config, task=task)