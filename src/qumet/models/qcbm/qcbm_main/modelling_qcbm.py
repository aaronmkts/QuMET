from typing import Dict
from pennylane.qnn import TorchLayer as TorchConnector
from pennylane import numpy as np
import torch.jit
import torch.nn as nn
import pennylane as qml
from torch import Tensor
from logging import getLogger

logger = getLogger(__name__)

config = {
    "generator": {
        "device": "default.qubit",
        "n_qubits": 9,
        "shots": 1000,
        "depth": 6,
        "diff_method": "backprop",
    },
}


class QCBMGenerator(nn.Module):
    def __init__(self, config, task):
        super().__init__()

        name = "generator"
        generator_config = config[name]

        self.depth = generator_config["depth"]
        self.n_qubits = generator_config["n_qubits"]
        self.device = generator_config["device"]
        self.shots = generator_config["shots"]
        self.diff_method = generator_config["diff_method"]
        self.q_layer = self._construct_quantum_layer()

    def _construct_quantum_layer(self):

        wshape = qml.StronglyEntanglingLayers.shape(n_layers=self.depth, n_wires=self.n_qubits)
        weights = np.random.random(size=wshape)
        self.weights = nn.Parameter(torch.tensor(weights, requires_grad=True, dtype=torch.float64))

        self.q_device = qml.device(self.device, wires=self.n_qubits)
        @qml.qnode(self.q_device, interface='torch', diff_method=self.diff_method)  
        def circuit(weights):

            qml.StronglyEntanglingLayers(
                weights=weights, ranges=[1] * self.depth, wires=range(self.n_qubits)
            )
            return qml.probs()

        return circuit

    def forward(self):

        circuit = self.q_layer
        prob_distribution = circuit(self.weights)

        return prob_distribution

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
        
    def forward(self):
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
