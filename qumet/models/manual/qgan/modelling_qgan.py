"""Pytorch & Pennylane Hybrid Qgan model"""

# Library imports
import math
import random
import numpy as np
import pennylane as qml
from pennylane.templates import AngleEmbedding
import sys
from torch import Tensor

# Pytorch imports
import torch
import torch.nn as nn
from typing import Optional, Union
from pennylane.qnn import TorchLayer as TorchConnector
from .configuration_qgan import QganConfig, QmlMixin

_CONFIG_FOR_DOC = "QganConfig"


class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(1, 64),
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


class Generator(nn.Module, QmlMixin):
    def __init__(
        self,
        config: QganConfig = QganConfig,
        n_qubits: int = 4,
        depth: int = 4,
        device: Optional[Union[str, qml.Device]] = "default.qubit",
    ) -> None:
        super(Generator, self).__init__()
        self.config = config
        self.n_qubits = n_qubits
        self.depth = depth

        self._set_qml_device(device)

        q_weight_shapes = {"q_weights": (self.depth * self.n_qubits)}
        q_generator = qml.QNode(self._circuit, self.device, interface="torch")
        batch_q_circuit = qml.batch_input(q_generator, argnum = 0 )
        self.q_generator = TorchConnector(batch_q_circuit, q_weight_shapes)

    def __str__(self):
        return f"QuantumGenerator({self.n_qubits}) "


    def _circuit(self, inputs, q_weights):
        """Builds the circuit to be fed to the connector as a QML node"""
        self._embed_features(inputs)
        q_weights = q_weights.reshape(self.depth, self.n_qubits)
        # Repeated layer
        for i in range(self.depth):
            for y in range(self.n_qubits):
                qml.RY(q_weights[i][y], wires=y)
            for y in range(self.n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.expval(qml.PauliZ(0))

    def _embed_features(self, features):
        wires = range(self.n_qubits)
        AngleEmbedding(features, wires=wires, rotation="X")

    def forward(self, noise: Tensor):
        return self.q_generator(noise)
