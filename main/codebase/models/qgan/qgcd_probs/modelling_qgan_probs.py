"""Pytorch & Pennylane Hybrid Qgan model"""

# Library imports
import math
import random
import numpy as np
import pennylane as qml
from pennylane.templates import AngleEmbedding
from torch import Tensor

# Pytorch imports
import torch
import torch.nn as nn
from typing import Optional, Union
from pennylane.qnn import TorchLayer as TorchConnector
from typing import Any, Callable, Dict, List, Optional, Type, Union

pi = math.pi

#fmt:0ff
config = {
    "discriminator":{
        "input_size": 2
    },
    "generator":{
        "device": "default.qubit",
        "n_qubits": 6,
        "n_a_qubits": 0,
        "shots": 10000,
        "depth": 4,
        "q_delta": 1,
        "diff_method": "adjoint",
    }
}

#fmt:on
class Probs_Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config, task):
        super().__init__()
        name = "discriminator"
        self.input_size = config[name]["input_size"]

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(self.input_size, 64),
            nn.LeakyReLU(),
            # First hidden layer (64 -> 64)
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

    def forward(self, input: Tensor)-> Tensor:
        return self.model(input)


class Probs_Generator(nn.Module):
    def __init__(
        self,
        config,
        task,
    ) -> None:
        super().__init__()

        name = "generator"
        self.n_qubits = config[name]["n_qubits"]
        self.depth = config[name]["depth"]
        self.shots = config[name]["shots"]
        self.device = config[name]["device"]

        self.device = qml.device(self.device, wires = self.n_qubits, shots = self.shots)
        q_weight_shapes = {"q_weights_y": (self.depth, self.n_qubits),
                           "q_weights_z": (self.depth, self.n_qubits)}
        init_method = {"q_weights_y": lambda x : torch.nn.init.uniform_(x, -pi, pi),
                        "q_weights_z": lambda x : torch.nn.init.uniform_(x, -pi, pi)}
        q_generator = qml.QNode(self._circuit, self.device, interface="torch")
        self.q_generator = TorchConnector(q_generator, q_weight_shapes, init_method = init_method)

    def __str__(self):
        return f"QuantumGenerator({self.n_qubits}) "


    def _circuit(self, inputs, q_weights_y, q_weights_z):
        """Builds the circuit to be fed to the connector as a QML node"""
        
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



class QGCD_Probs_GAN(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks
        
        self.generator = Probs_Generator(config, task)
        self.discriminator = Probs_Discriminator(config, task)
        
    def forward(self, input: Tensor):
        return self.generator(input)
    

# ---------------------------------------
# QGCD_GANs
# ---------------------------------------
    
def _qgcd_gan(config, task : str) -> QGCD_Probs_GAN:

    model = QGCD_Probs_GAN(config, task)
    return model
    
def get_qgcd_probs(info: Dict) -> QGCD_Probs_GAN:

    task = info.generation
    return _qgcd_gan(config = config,
                     task = task)