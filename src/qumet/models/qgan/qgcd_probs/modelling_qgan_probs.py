"""Pytorch & Pennylane Hybrid Qgan model"""

# Library imports
import math
import random
import numpy as np
from torch import Tensor
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import partial
import matplotlib.pyplot as plt
# Pytorch & Pennylaneimports
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
    "discriminator": {"input_size": 2},
    "generator": {
        "device": "default.qubit",
        "n_qubits": 6,
        "n_a_qubits": 0,
        "shots": 10000,
        "depth": 2,
        "q_delta": 1,
        "diff_method": "adjoint",
    },
}


def wires_pairwise(wires):
    """Wire sequence for the pairwise pattern."""
    sequence = []
    for layer in range(2):
        block = wires[layer : len(wires) - layer]
                
        sequence += [block.subset([i, i + 1]) for i in range(0, len(block) - 1, 2)]
    return sequence

# fmt:on
class Probs_Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config, task):
        super(Probs_Discriminator, self).__init__()
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
        
        self.device = qml.device(self.device,
                                  wires=self.n_qubits, 
                                  shots=self.shots)
        
        q_weight_shapes = {
            "q_weights_0": (self.n_qubits, 2),
            "q_weights_i": (self.depth, self.n_qubits, 2),
        }
        init_method = {
            "q_weights_0": lambda x: torch.nn.init.uniform_(x, -pi, pi),
            "q_weights_i": lambda x: torch.nn.init.uniform_(x, -pi, pi),
        }

        self.q_generator = self._make_qnode(q_weight_shapes, init_method)
        
    def _visualise(self):

        inputs = torch.tensor([])
        fig, ax = qml.draw_mpl(self.q_generator, style = 'pennylane')(inputs)
        fig.show()
        return plt.show()

    def _make_qnode(self, q_weight_shapes, init_method):
        wires = list(range(self.n_qubits))
        
        #NOTE: Qnode is composed of alternating layers of rotation gates and engangling gates.
        #This implementation is for the Efficient SU(2) circuit with pairwise entanglement & (RY,RX) rotations.

        def __angle_layer(q_weights, wires):
            
            def template(q_weights_y,q_weights_z,  wires):
                qml.RY(q_weights_y, wires = wires)
                qml.RZ(q_weights_z, wires = wires)

            broadcast(unitary = template, pattern = 'single', wires = wires, parameters=q_weights)
   
        def __entangling_layer(entangler: str, pattern: str, wires:int):
            match entangler:
                case 'CNOT':
                    entangling_operation = qml.CNOT
                case 'CZ':
                    entangling_operation = qml.CZ

            broadcast(unitary= entangling_operation, pattern = pattern, wires = wires) 

        @qml.qnode(self.device,  interface='torch')
        def _qnode(inputs, q_weights_0, q_weights_i):
            """Builds the circuit to be fed to the connector as a QML node"""

            sequence = wires_pairwise(Wires(wires))
            __angle_layer(q_weights_0, wires)
        
            def _subroutine(q_weights_i, wires):
                qml.Barrier(wires)
                __entangling_layer(entangler = 'CNOT', pattern = sequence, wires = wires)
                qml.Barrier(wires)
                __angle_layer(q_weights_i, wires)
                
            
            qml.layer(_subroutine, self.depth, q_weights_i, wires = wires)

            return qml.probs()

        return TorchConnector(_qnode, q_weight_shapes, init_method = init_method)

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


def _qgcd_gan(config, task: str) -> QGCD_Probs_GAN:

    model = QGCD_Probs_GAN(config, task)
    return model


def get_qgcd_probs(info: Dict) -> QGCD_Probs_GAN:

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into QGCD_PROBS_GAN ")
    return _qgcd_gan(config=config, task=task)
