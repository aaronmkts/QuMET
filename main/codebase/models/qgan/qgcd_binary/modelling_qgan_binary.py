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

from typing import Callable, Sequence, Union
import copy
import numpy as np
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.core import transform
from pennylane.transforms.batch_params import _nested_stack


@transform
def get_probslist(tape: QuantumTape) -> tuple[Sequence[QuantumTape], Callable]:

    tape = copy.deepcopy(tape)

    measurement = qml.probs()
    shots = 10000

    new_tape = QuantumScript(
        tape.operations,
        measurements=measurement,
        shots=shots,
        trainable_params=tape.trainable_params,
    )

    def processing_fn(res):
        return res[0]

    return [new_tape], processing_fn


@transform
def batch_input(
    tape: QuantumTape,
    argnum: Union[Sequence[int], int],
) -> (Sequence[QuantumTape], Callable):
    batch_size = argnum

    all_parameters = tape.get_parameters(trainable_only=False)

    output_tapes = []

    def _split_operations(ops, params, num_tapes):

        new_ops = [[] for _ in range(num_tapes)]

        for op in ops:
            for b in range(num_tapes):
                new_ops[b].append(op)

        return new_ops

    for ops in _split_operations(tape.operations, all_parameters, batch_size):
        new_tape = QuantumScript(
            ops,
            tape.measurements,
            shots=tape.shots,
            trainable_params=tape.trainable_params,
        )
        output_tapes.append(new_tape)

    def processing_fn(res):
        return _nested_stack(res)

    return output_tapes, processing_fn


_CONFIG_FOR_DOC = "QGCD_Binary_Config"

pi = math.pi


class Binary_Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config: QGCD_Binary_Config):
        super(Binary_Discriminator, self).__init__()
        self.input_size = config.input_size

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(self.input_size, 64),
            nn.LeakyReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            # Second hidden layer (64 -> 64)
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            # Third hidden layer (64 -> num_output_features)
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # self.model.apply(self.weights_init)

    def weights_init(self, layer):
        classname = layer.__class__.__name__
        if classname.find("Linear") != -1:
            torch.nn.init.kaiming_uniform_(layer.weight.data, 0.0, 0.02)

    def forward(self, input):
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
        self.diff_method = config.diff_method

        self.device = qml.device(
            config.device, wires=config.n_qubits, shots=config.shots
        )

        q_weight_shapes = {
            "q_weights_y": (self.depth, self.n_qubits),
            "q_weights_z": (self.depth, self.n_qubits),
        }

        init_method = {
            "q_weights_y": lambda x: torch.nn.init.uniform_(x, -pi, pi),
            "q_weights_z": lambda x: torch.nn.init.uniform_(x, -pi, pi),
        }

        self.q_generator = qml.QNode(
            self._circuit, device=self.device, interface="torch"
        )

        self.batch_q_circuit = qml.batch_input(self.q_generator, argnum=0)
        self.batch_q_generator = TorchConnector(
            self.batch_q_circuit, q_weight_shapes, init_method=init_method
        )

    def __str__(self):
        return f"QuantumGenerator({self.n_qubits}) "

    def _get_batchdim(self, inputs):
        batch_dims = np.unique(inputs)
        if len(batch_dims) != 1:
            raise ValueError("Batch dimension for all gate arguments must be the same.")

        batch_size = batch_dims[0]

        return batch_size

    def _embed_features(self, features):
        wires = range(self.n_qubits)
        AngleEmbedding(features, wires=wires, rotation="X")

    def _circuit(self, inputs, q_weights_y, q_weights_z):
        """Builds the circuit to be fed to the connector as a QML node"""
        # Embedding layer
        self._embed_features(inputs)
        # Repeated layer

        for i in range(self.depth):
            for y in range(self.n_qubits):
                qml.RY(q_weights_y[i][y], wires=y)
                qml.RZ(q_weights_z[i][y], wires=y)
            for y in range(self.n_qubits - 1):
                qml.CNOT(wires=[y, y + 1])

        return qml.sample()

    def forward(self, inputs: Tensor):
        return self.batch_q_generator(inputs)
