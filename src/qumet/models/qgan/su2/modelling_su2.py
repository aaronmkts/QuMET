from typing import Dict
from pennylane.qnn import TorchLayer as TorchConnector
from pennylane import numpy as np
import torch.jit
import torch.nn as nn
import pennylane as qml
from torch import Tensor
from logging import getLogger
from pennylane import broadcast
from pennylane.wires import Wires
from typing import Optional as _Optional
import warnings
import torch.overrides
from torch.nn.init import calculate_gain, _calculate_correct_fan
from copy import deepcopy
import math 

logger = getLogger(__name__)


n_qubits = 8
config = {
    "discriminator": {
        "input_size": 2 ** n_qubits,
    },
    "generator": {
        "device": "default.qubit",
        "n_qubits": n_qubits,
        "shots": 10000,
        "depth": 6,
        "diff_method": "parameter-shift",
    },
}

pi = math.pi 



#Discriminator
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
        return self.model(input)

class SU2Generator(nn.Module):
    def __init__(self, config, task):
        super().__init__()

        name = "generator"
        self.n_qubits = config[name]["n_qubits"]
        self.depth = config[name]["depth"]
        self.shots = config[name]["shots"]
        self.device = config[name]["device"]
        self.diff_method = config[name]["diff_method"]

        self.q_layer = self._construct_quantum_layer()

    def _wires_pairwise(self, wires):
        """Wire sequence for the pairwise pattern."""
        sequence = []
        for layer in range(2):
            block = wires[layer : len(wires) - layer]
                    
            sequence += [block.subset([i, i + 1]) for i in range(0, len(block) - 1, 2)]
        return sequence

    def _entangling_layer(self, entangler: str, pattern: str, wires:int):
        match entangler:
            case 'CNOT':
                entangling_operation = qml.CNOT
            case 'CZ':
                entangling_operation = qml.CZ

        broadcast(unitary= entangling_operation, pattern = pattern, wires = wires) 

    def _angle_layer(self, q_weights, wires):
            
        def template(q_weights_y,q_weights_z,  wires):
            qml.RY(q_weights_y, wires = wires)
            qml.RZ(q_weights_z, wires = wires)

        broadcast(unitary = template, pattern = 'single', wires = wires, parameters=q_weights)

    def _construct_quantum_layer(self):
        wires = list(range(self.n_qubits))


        weights = np.random.random(size=(self.n_qubits,2))
        self.weights_0 = nn.Parameter(torch.tensor(weights, requires_grad=True, dtype=torch.float32))

        weights = np.random.random(size=(self.depth, self.n_qubits, 2))
        self.weights_i = nn.Parameter(torch.tensor(weights, requires_grad=True, dtype=torch.float32))

        self.q_device = qml.device(self.device, wires=self.n_qubits, shots=self.shots)
        @qml.qnode(self.q_device,  interface='torch', diff_method=self.diff_method)
        def circuit(q_weights_0, q_weights_i):
            """Builds the circuit to be fed to the connector as a QML node"""
            
            sequence = self._wires_pairwise(Wires(wires))
            self._angle_layer(q_weights_0, wires)
        
            def _subroutine(q_weights_i, wires):
                qml.Barrier(wires)
                self._entangling_layer(entangler = 'CNOT', pattern = sequence, wires = wires)
                qml.Barrier(wires)
                self._angle_layer(q_weights_i, wires)
                
            
            qml.layer(_subroutine, self.depth, q_weights_i, wires = wires)

            return qml.probs()
        
        return circuit
    
    def forward(self):
        circuit = self.q_layer
        prob_distribution = circuit(self.weights_0, self.weights_i)

        return prob_distribution


class SU2(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks
        self.generator = SU2Generator(config, task)
        self.discriminator = Probs_Discriminator(config, task)

    def forward(self):
        return self.generator()


# ---------------------------------------
# QCBM
# ---------------------------------------


def _su2(config, task: str) -> SU2:
    model = SU2(config, task)
    return model


def get_su2(info: Dict) -> SU2:
    task = "info.generation"
    logger.info(f"The following {config} loaded for task into QCBM")
    return _su2(config=config, task=task)
