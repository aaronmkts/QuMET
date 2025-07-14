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
#from pennylane import broadcast
from pennylane.wires import Wires


logger = getLogger(__name__)
pi = math.pi

# fmt:0ff
config = {
    "discriminator": {"input_size": 8}, #Due to image resize, height/width is 8
    "generator": {
        "device": "default.qubit",
        "n_qubits": 5,
        "n_a_qubits": 1,
        "shots": 10000,
        "depth": 6,
        "q_delta": 1,
        "diff_method": "best",
        "diff_method": "best",
        "n_generators": 4,
        "q_delta": 1

    },
}

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


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, config, task):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """
        name = "generator"
        generator_config = config[name]
        self.n_generators = generator_config["n_generators"]
        self.device = generator_config["device"]
        self.n_qubits = generator_config["n_qubits"]
        self.n_a_qubits = generator_config["n_a_qubits"]
        self.depth = generator_config["depth"]
        self.diff_method = generator_config["diff_method"]
        self.q_device = qml.device(self.device, wires= self.n_qubits)
        self._construct_quantum_layers()
   
    def _construct_quantum_layers(self):
        qnode = qml.QNode(self.circuit, self.q_device, interface="torch", diff_method=self.diff_method)
        weight_shapes = {"weights": (self.depth, self.n_qubits)}
        self.q_layers = nn.ModuleList(
            [TorchConnector(qnode, weight_shapes) for _ in range(self.n_generators)]
        )

    def partial_measure(self,noise, q_layer):
        # Non-linear Transform
        probs = q_layer(noise)
        probs = q_layer(noise)
        probsgiven0 = probs[: (2 ** (self.n_qubits - self.n_a_qubits))]
        probsgiven0 /= torch.sum(probs)

        # Post-Processing
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven

    def circuit(self, inputs, weights):
        weights = weights.reshape(self.depth, self.n_qubits)

        # Initialise latent vectors
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RY(inputs[i], wires=i)

        # Repeated layer
        for i in range(self.depth):
            # Parameterised layer
            for y in range(self.n_qubits):
                qml.RY(weights[i][y], wires=y)

            # Control Z gates
            for y in range(self.n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(self.n_qubits)))
    
    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (self.n_qubits - self.n_a_qubits)
        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0)
        # Iterate over all sub-generators

        for q_layer in self.q_layers:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size)
            # for b in batch basically
            for elem in x:
                q_out = self.partial_measure(elem, q_layer).float().unsqueeze(0)
                q_out = self.partial_measure(elem, q_layer).float().unsqueeze(0)
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

    def forward(self, z: Tensor):
        with torch.no_grad():
            imgs_batch = self.generator(z)
            imgs = imgs_batch.view(-1, 1, 8, 8)
        return imgs

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