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
from typing import Optional as _Optional
import warnings
import torch.overrides
from torch.nn.init import calculate_gain, _calculate_correct_fan
from copy import deepcopy
from itertools import chain

logger = getLogger(__name__)
pi = math.pi

# fmt:0ff
n_qubits = 10
config = {
    "discriminator": {"input_size": n_qubits},
    "generator": {
        "device": "lightning.qubit",
        "n_qubits": n_qubits,
        "n_a_qubits": 0,
        "shots": 16,
        "depth": 2,
        "q_delta": 1,
        "diff_method": "None",
    },
}

# fmt:on

#Discriminator
class Bitstring_Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config, task):
        super(Bitstring_Discriminator, self).__init__()
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
            nn.Sigmoid()
        )

        self.model.apply(self.init_weights)

    def alt_kaiming_uniform_(self, tensor: torch.Tensor,
        a: float = 0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
        generator: _Optional[torch.Generator] = None,
    ):
        if torch.overrides.has_torch_function_variadic(tensor):
            return torch.overrides.handle_torch_function(
                self.alt_kaiming_uniform_,
                (tensor,),
                tensor=tensor,
                a=a,
                mode=mode,
                nonlinearity=nonlinearity,
                generator=generator)

        if 0 in tensor.shape:
            warnings.warn("Initializing zero-element tensors is a no-op")
            return tensor
        fan = _calculate_correct_fan(tensor, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        std = 2*std
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            return tensor.uniform_(-bound, bound, generator=generator)
        
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            self.alt_kaiming_uniform_(layer.weight, 
                                     mode = 'fan_out', 
                                     a = math.sqrt(5))
           
    def forward(self, input: Tensor) -> Tensor:

        return self.model(input).view(-1)


name = 'generator'

n_qubits = config[name]["n_qubits"]
wires = list(range(n_qubits))
device = config[name]["device"]
dev = qml.device(device, wires = n_qubits)
depth = config[name]["depth"]

#Generator
def wires_pairwise(wires):
    """Wire sequence for the pairwise pattern."""
    sequence = []
    for layer in range(2):
        block = wires[layer : len(wires) - layer]
                
        sequence += [block.subset([i, i + 1]) for i in range(0, len(block) - 1, 2)]
    return sequence

def __angle_layer(q_weights, wires):
            
    def template(q_weights_y, q_weights_x,  wires):
        qml.RY(q_weights_y, wires = wires)
        qml.RX(q_weights_x, wires = wires)

    broadcast(unitary = template, pattern = 'single', wires = wires, parameters=q_weights)

def __entangling_layer(entangler: str, pattern: str, wires:int):
    match entangler:
        case 'CNOT':
            entangling_operation = qml.CNOT
        case 'CZ':
            entangling_operation = qml.CZ

    broadcast(unitary= entangling_operation, pattern = pattern, wires = wires) 

def _subroutine(q_weights_i, wires):
        qml.Barrier(wires)
        __entangling_layer(entangler = 'CNOT', pattern = wires_pairwise(Wires(wires)), wires = wires)
        qml.Barrier(wires)
        __angle_layer(q_weights_i, wires)


@qml.qnode(dev, interface='torch')
def circuit(weights, mode):
    """Builds the circuit to be fed to the connector as a QML node"""

    __angle_layer(weights[0], wires)
    qml.layer(_subroutine, depth, weights[1:], wires = wires)
    if mode == 'train':
        return qml.sample()
    elif mode == 'evaluate':
        return qml.probs()



class EfficientSU2(nn.Module):
    def __init__(self, config, task):
        super(EfficientSU2, self).__init__()
        name = 'generator'
        self.n_qubits = config[name]["n_qubits"]
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(nn.init.uniform_(torch.empty((depth+1, n_qubits, 2), dtype=torch.float32), a = -pi, b = pi),
                              requires_grad=True)
            ]
        )
        self.shifts = self._preprocess_shifts()

    def _preprocess_shifts(self):
        copy_params = deepcopy(self.q_params)
        flat_params = copy_params[0].flatten()
        
        elementary_shift = pi / 2
        shifts = [
            [elementary_shift * e_i, -elementary_shift * e_i]
            for e_i in torch.eye(*flat_params.size())
        ]
        
        shifts = list(chain(*shifts))
        shifts = [shift.reshape(*flat_params.size()) for shift in shifts]
        shifts = [shift.reshape(*self.q_params[0].size()) for shift in shifts]
      
        return shifts
    
    def get_gradient(self, shots, discriminator):
        mode = "train"
        def criterion(outputs):
            return torch.mean((-1.0 * torch.log(outputs)))
                              
        q_weights_0 = self.q_params[0].detach()

        generator_samples = [sample.type(torch.float32) for sample in [circuit(
            q_weights_0 + parameter_shift, mode, shots = shots
        ) for parameter_shift in self.shifts]]
    
        gradients = []
        for i in range(0, len(generator_samples), 2):
            forward_fake = generator_samples[i]
            backward_fake = generator_samples[i + 1]
            
            forward_output = discriminator(forward_fake).flatten().detach()
            backward_output = discriminator(backward_fake).flatten().detach()

            forward_diff = criterion(forward_output)
            backward_diff = criterion(backward_output)
            gradients.append(1 / 2 * (forward_diff - backward_diff))
            
        gradients = torch.reshape(torch.tensor(gradients), q_weights_0.size())
        return gradients

    def forward(self, shots, mode):
        q_weights_0 = self.q_params[0]

        if mode == 'train':
            result = circuit(q_weights_0, mode, shots = shots)
            result = result.type(torch.float32)
        elif mode == 'evaluate':
            result = circuit(q_weights_0, mode, shots = shots)
        return result

class QGCD_EfficientSU2(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks

        self.generator = EfficientSU2(config, task)
        self.discriminator = Bitstring_Discriminator(config, task)

    def forward(self, input: Tensor):
        return self.generator(input)


# ---------------------------------------
# QGCD_GANs
# ---------------------------------------


def _qgcd_gan(config, task: str) -> QGCD_EfficientSU2:

    model = QGCD_EfficientSU2(config, task)
    return model


def get_qgcd_bitstring(info: Dict) -> QGCD_EfficientSU2:

    task = "bitstring_generation"
    logger.info(f"The following {config} loaded for task {task} into EfficientSU2 QGAN ")
    return _qgcd_gan(config=config, task=task)
