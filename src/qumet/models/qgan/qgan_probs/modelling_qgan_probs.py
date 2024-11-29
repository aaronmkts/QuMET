"""Pytorch & Pennylane Hybrid Qgan model"""

# Library imports
import math
import random
import numpy as np
from torch import Tensor
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Pytorch & Pennylaneimports
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer as TorchConnector
from pennylane.wires import Wires

logger = getLogger(__name__)
pi = math.pi

# fmt:0ff
config = {
    "generator": {
        "device": "default.qubit",
        "n_qubits": 10,
        "shots": 10000,
        "depth": 3,
        "q_delta": 1,
        "diff_method": "best",
        "q_delta": 1,
    },
    "discriminator": {
        'input_size': 1024
        }, 
}

class QuantumGenerator(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        name = "generator"
        generator_config = config[name] 

        self.n_qubits = generator_config["n_qubits"]
        self.depth = generator_config["depth"]
        self.shots = generator_config["shots"]
        self.device = generator_config["device"]
        self.diff_method = generator_config["diff_method"]
        self.q_device = qml.device(self.device,
                                  wires=self.n_qubits, 
                                  shots=self.shots)
        self._construct_quantum_layer()
        
    def _construct_quantum_layer(self):
        qnode = qml.QNode(self.circuit, self.q_device, interface="torch", diff_method=self.diff_method)
        weight_shapes = {"weights": (self.depth, self.n_qubits, 3)}
        self.q_layer = TorchConnector(qnode, weight_shapes)

    def circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        for i in range(self.depth):
            for j in range(self.n_qubits):
                qml.Rot(*weights[i][j], wires=j)

            for j in range(self.n_qubits-1):
                qml.CNOT(wires=[j, j+1])
                
            qml.CNOT(wires=[self.n_qubits-1, 0])
        
        return qml.probs(wires=list(range(self.n_qubits)))

    def forward(self, inputs: Tensor):
        # Handle batched inputs by iterating over the batch dimension
        batch_outputs = []
        for i in range(inputs.shape[0]):  # Iterate over batch dimension
            batch_outputs.append(self.q_layer(inputs[i]))  # Apply q_layer to each item in the batch

        return torch.stack(batch_outputs) 


class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config):
        super(Discriminator, self).__init__()
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
            # Third hidden layer (64 -> num_output_features)
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.model.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, 
                                     mode = 'fan_out', 
                                     a = 2 * math.sqrt(5))
    
    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)



class QGCD_Probs_GAN(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks
      
        self.generator = QuantumGenerator(config)
        self.discriminator = Discriminator(config)

    def forward(self, input: Tensor):
        return self.generator(input)


# ---------------------------------------
# QGCD_GANs
# ---------------------------------------


def _qgcd_gan(config, task: str) -> QGCD_Probs_GAN:

    model = QGCD_Probs_GAN(config, task)
    return model


def get_qgan_probs(info: Dict) -> QGCD_Probs_GAN:

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into QGCD_PROBS_GAN ")
    return _qgcd_gan(config=config, task=task)
