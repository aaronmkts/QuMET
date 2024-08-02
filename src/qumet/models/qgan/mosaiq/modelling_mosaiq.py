"""
Pytorch & Pennylane Hybrid Qgan model
Code Adapted from https://github.com/SilverEngineered/MosaiQ/blob/main/mosaiq.py
"""

# Library imports
import math
from torch import Tensor
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import partial
# Pytorch & Pennylane imports
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer as TorchConnector


logger = getLogger(__name__)
pi = math.pi

# fmt:0ff
pca_dims = 40
image_size = 5
config = {
    "discriminator": {"input_size": pca_dims},
    "generator": {
        "device": "default.qubit",
        "n_qubits": 5,
        "n_a_qubits": 0,
        "shots": 10000,
        "depth": 6,
        "q_delta": 1,
        "diff_method": "best",
        "n_generators": 8,
        "q_delta": 1,
        "pcs_dims": pca_dims
    },
}
# fmt:on

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
    

class MosaiQGenerator(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        
        name = "generator"
        generator_config = config[name]

        q_delta = config[name]["q_delta"]

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

    def circuit(self, inputs, weights):
        weights = weights.reshape(self.depth, self.n_qubits)
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RX(inputs[i], wires=i)
        for i in range(self.depth):
            for y in range(self.n_qubits):
                qml.RY(weights[i][y], wires=y)
            for y in range(self.n_qubits - 1):
                qml.CZ(wires=[y, y + 1])
        return [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)]
    
    def feature_redistribution(self):
        ordering = []
        for i in range(8):
            k = 4* i
            l = [i, 39-k, 38-k, 37-k, 36-k]
            ordering.append(l)
        return ordering

    def forward(self, x):
        batch_size = x.size(0)
        images = []

        patch_size = image_size
        ordering = self.feature_redistribution()
        images = torch.Tensor(x.size(0), 0)
        for q_layer in self.q_layers:
            patches = torch.Tensor(0, patch_size)
            for elem in x:
                f = q_layer(elem)
                f = tuple(f)
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