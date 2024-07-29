from typing import Dict

import numpy as np
import torch.jit
import torch.nn as nn
import pennylane as qml
from torch import Tensor
from logging import getLogger

logger = getLogger(__name__)

config = {
    "generator": {
        "device": "lightning.qubit",
        "n_qubits": 9,
        "n_a_qubits": 1,
        "shots": 10000,
        "depth": 6,
        "q_delta": 1,
        "diff_method": "adjoint",
        "q_delta": 1

    },
}

# GENERATOR
name = "generator"
generator_config = config[name]

device = "default.qubit"  # generator_config["device"]
n_qubits = generator_config["n_qubits"]
n_a_qubits = generator_config["n_a_qubits"]
depth = generator_config["depth"]  # n_layers
dev = qml.device(device, wires=n_qubits, shots=generator_config['shots'])


@qml.qnode(dev, interface='torch', diff_method="parameter-shift")
def circuit(weights):
    weights = weights.repeat(1, 3).reshape(depth, n_qubits, 3)
    qml.StronglyEntanglingLayers(
        weights=weights, ranges=[1] * depth, wires=range(n_qubits)
    )
    return qml.probs()

class QCBMGenerator(nn.Module):
    def __init__(self, config, task):
        name = "generator"
        q_delta = config[name]["q_delta"]
        self.n_qubits = generator_config["n_qubits"]
        super().__init__()

        self.circ = circuit
        self.weights = torch.tensor(np.random.random(size=(depth, n_qubits)), requires_grad=True, dtype=torch.float64)
        self.params = nn.ParameterList([nn.Parameter(w.unsqueeze(0)) for w in self.weights])
    def forward(self):
        return self.circ(self.weights) #px

    def kl_divergence(self, px, py):
        qcbm_probs = px.clone().detach()
        target_probs = py
        kl_div = -torch.sum(target_probs * torch.nan_to_num(torch.log(qcbm_probs / target_probs)))
        return kl_div


class QCBM(nn.Module):
    def __init__(self, config, task):
        super().__init__()
        # networks
        self.generator = QCBMGenerator(config, task)

    def forward(self, input: Tensor=None):
        return self.generator()


# ---------------------------------------
# QCBM
# ---------------------------------------


def _qcbm(config, task: str) -> QCBM:
    model = QCBM(config, task)
    return model


def get_qcbm(info: Dict) -> QCBM:
    task = "info.generation"
    logger.info(f"The following {config} loaded for task into QCBM")
    return _qcbm(config=config, task=task)
