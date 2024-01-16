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
from .configuration_qgan_binary import QganConfig, QmlMixin

_CONFIG_FOR_DOC = "QganConfig"


class Binary_Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        n_qubits: int = 4
        super(Binary_Discriminator,self).__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(n_qubits, 64),
            nn.LeakyReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            # Second hidden layer (64 -> 64)
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            #Third hidden layer (64 -> num_output_features)
            nn.Linear(64, 1),
            nn.LogSigmoid(),
        )

        self.model.apply(self.init_weights)
    
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight, mode = 'fan_out')
            layer.bias.fill_(0.01)

    def forward(self, x):
        return self.model(x)


class Binary_Generator(nn.Module, QmlMixin):
    def __init__(
        self,
        config: QganConfig = QganConfig,
        n_qubits: int = 4,
        depth: int = 3,
        device: Optional[Union[str, qml.Device]] = "default.qubit",
    ) -> None:
        super(Binary_Generator, self).__init__()
        self.config = config
        self.n_qubits = n_qubits
        self.depth = depth

        self._set_qml_device(device)

        q_weight_shapes = {"q_weights_y": (self.depth * self.n_qubits),
                           "q_weights_z": ((self.depth * self.n_qubits)) }
        q_generator = qml.QNode(self._circuit, self.device, interface="torch")
        self.q_generator = TorchConnector(q_generator, q_weight_shapes, init_method = nn.init.uniform_(-math.pi, math.pi))

    def __str__(self):
        return f"QuantumGenerator({self.n_qubits}) "


    def _circuit(self, q_weights_y, q_weights_z):
        """Builds the circuit to be fed to the connector as a QML node"""
        q_weights_y = q_weights_y.reshape(self.depth, self.n_qubits)
        q_weights_z = q_weights_z.reshape(self.depth, self.n_qubits)
        # Repeated layer
        for i in range(self.depth):
            for y in range(self.n_qubits):
                qml.RY(q_weights_y[i][y], wires = y)
                qml.RZ(q_weights_z[i][y], wires = y)
            for y in range(self.n_qubits - 1):
                qml.CNOT(wires=[y, y + 1])

        return qml.sample(qml.PauliZ(0))

    def forward(self, noise: Tensor):
        return self.q_generator(noise)
