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

torch.manual_seed(42)

logger = getLogger(__name__)
pi = math.pi

# fmt:0ff
n_qubits = 6
config = {
    "discriminator": {"input_size": n_qubits},
    "generator": {
        "device": "lightning.qubit",
        "n_qubits": n_qubits,
        "n_a_qubits": 0,
        "shots": 16,
        "depth": 2,
        "q_delta": 1,
        "diff_method": "parameter-shift",
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
        output = self.model(input)
        output = output.view(-1)
        return output

class Bitstring_Generator(nn.Module):
    def __init__(
        self,
        config,
        task,
    ) -> None:
        super().__init__()

        name = "generator"
        generator_config = config[name]
        
        self.n_qubits = generator_config["n_qubits"]
        self.depth = generator_config["depth"]
        self.diff_method = generator_config["diff_method"]
        self.device = generator_config["device"]
        
        self.device = qml.device(self.device,
                                  wires=self.n_qubits)
        


        self.q_weights_0 = nn.init.uniform_(torch.empty((self.n_qubits, 2), requires_grad=True, dtype=torch.float32), a= -pi, b = pi)
        self.q_weights_i = nn.init.uniform_(torch.empty((self.depth, self.n_qubits, 2), requires_grad=True, dtype=torch.float32), a = -pi, b = pi)
        self.q_weights_0.retain_grad()
        self.q_weights_i.retain_grad()
        self.parameters = [self.q_weights_0, self.q_weights_i]
        
    
    def _visualise(self):

        inputs = torch.tensor([])
        fig, ax = qml.draw_mpl(self.q_generator, style = 'pennylane')(inputs)
        fig.show()

        return plt.show()

    def _make_qnode(self, q_weights_0, q_weights_i, batch_size, measure):
        wires = list(range(self.n_qubits))
        shots = [1] * batch_size if measure == 0 else batch_size

        #NOTE: Qnode is composed of alternating layers of rotation gates and engangling gates.
        #This implementation is for the Efficient SU(2) circuit with pairwise entanglement & (RY,RX) rotations.
        
        def __angle_layer(q_weights, wires):
            
            def template(q_weights_y,q_weights_x,  wires):
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

        @qml.qnode(self.device, 
                   diff_method = self.diff_method, 
                   interface='torch')
        def _qnode(q_weights_0, q_weights_i, measure):
            """Builds the circuit to be fed to the connector as a QML node"""

            sequence = wires_pairwise(Wires(wires))
            __angle_layer(q_weights_0, wires)
        
            def _subroutine(q_weights_i, wires):
                qml.Barrier(wires)
                __entangling_layer(entangler = 'CNOT', pattern = sequence, wires = wires)
                qml.Barrier(wires)
                __angle_layer(q_weights_i, wires)
                
            
            qml.layer(_subroutine, self.depth, q_weights_i, wires = wires)
            if measure == 0 :
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            else:
                return qml.probs()

        return _qnode(q_weights_0, q_weights_i, measure, shots = shots)

    def forward(self, batch_size, measure):
        if measure == 0:     

            samples = self._make_qnode(self.q_weights_0, self.q_weights_i, batch_size, measure)
            samples = (torch.stack([torch.stack(s) for s in samples])+1) / 2
            output = samples.to(dtype=torch.float32)
        else:
            output = self._make_qnode(self.q_weights_0, self.q_weights_i, batch_size, measure)
        return output

class QGCD_Bitstring_GAN(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks

        self.generator = Bitstring_Generator(config, task)
        self.discriminator = Bitstring_Discriminator(config, task)

    def forward(self, input: Tensor):
        return self.generator(input)


# ---------------------------------------
# QGCD_GANs
# ---------------------------------------


def _qgcd_gan(config, task: str) -> QGCD_Bitstring_GAN:

    model = QGCD_Bitstring_GAN(config, task)
    return model


def get_qgcd_bitstring(info: Dict) -> QGCD_Bitstring_GAN:

    task = "bitstring_generation"
    logger.info(f"The following {config} loaded for task {task} into QGCD_PROBS_GAN ")
    return _qgcd_gan(config=config, task=task)
