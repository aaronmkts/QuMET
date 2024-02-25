"""Pytorch & Pennylane Hybrid QGAN model"""

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
from .configuration_qgcd import QGCDConfig, QmlMixin
from pytorch_lightning.core import LightningModule
_CONFIG_FOR_DOC = "QGCDConfig"


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
        config: QGCDConfig = QGCDConfig,
        n_qubits: int = 6,
        depth: int = 4,
        device: Optional[Union[str, qml.Device]] = "default.qubit",
    ) -> None:
        super(Generator, self).__init__()
        self.config = config
        self.n_qubits = n_qubits
        self.depth = depth

        self._set_qml_device(device)

        q_weight_shapes = {"q_weights": (self.depth * self.n_qubits)}
        self.q_generator = qml.QNode(self._circuit, self.device, interface="torch")
        batch_q_circuit = qml.batch_input(self.q_generator, argnum = 0)
        self.batch_q_generator = TorchConnector(batch_q_circuit, q_weight_shapes)

    def __str__(self):
        return f"QuantumGenerator({self.n_qubits}) "

    def _draw_circuit(self):
        noise = torch.rand(1, self.n_qubits) * math.pi / 2
        circuit = self.batch_q_generator
        drawer = qml.draw(circuit)
        return print(drawer(noise))

    def _circuit(self, inputs, q_weights):
        """Builds the circuit to be fed to the connector as a QML node"""
        self._embed_features(inputs)
        q_weights = q_weights.reshape(self.depth, self.n_qubits)
        # Repeated layer
        for i in range(self.depth):
            for y in range(self.n_qubits):
                qml.RY(q_weights[i][y], wires=y)
            for y in range(self.n_qubits - 1):
                qml.CNOT(wires=[y, y + 1])

        return qml.expval(qml.PauliZ(0))

    def _embed_features(self, features):
        wires = range(self.n_qubits)
        AngleEmbedding(features, wires=wires, rotation="X")

    def forward(self, 
                noise: Tensor):
        return self.batch_q_generator(noise)


class QGCD(nn.Module):
    def __init__(self, config: QGCDConfig):
        super().__init__(config)
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

        self.post_init()

        def forward(self,
                    noise: Tensor):
            return self.generator(noise)
