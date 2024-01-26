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
from .configuration_qgan_binary import QGCD_Binary_Config

_CONFIG_FOR_DOC = "QGCD_Binary_Config"

pi = math.pi

class Binary_Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config: QGCD_Binary_Config):
        super(Binary_Discriminator,self).__init__()
        self.input_size = config.n_qubits

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            # Second hidden layer (64 -> 64)
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            #Third hidden layer (64 -> num_output_features)
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        #self.model.apply(self.init_weights)
    
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight)
            torch.nn.init.kaiming_uniform_(layer.bias)

    def forward(self, 
                input):
        return self.model(input)


class Binary_Generator(nn.Module):
    def __init__(
        self,
        config: QGCD_Binary_Config,
    ) -> None:
        super(Binary_Generator, self).__init__()
        self.config = config
        self.n_qubits = config.n_qubits
        self.depth = config.depth
        self.device = qml.device(config.device, wires = config.n_qubits, shots = config.shots)
        q_weight_shapes = {"q_weights_y": (self.depth * self.n_qubits),
                           "q_weights_z": (self.depth * self.n_qubits)}
        init_method = {"q_weights_y": lambda x : torch.nn.init.uniform_(x, -pi, pi),
                        "q_weights_z": lambda x : torch.nn.init.uniform_(x, -pi, pi)}
        q_generator = qml.QNode(self._circuit, self.device, interface="torch")
        self.q_generator = TorchConnector(q_generator, q_weight_shapes, init_method = init_method)

    def __str__(self):
        return f"QuantumGenerator({self.n_qubits}) "


    def _circuit(self, inputs, q_weights_y, q_weights_z):
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

        return qml.probs()

    def forward(self, inputs: Tensor):
        return self.q_generator(inputs)
