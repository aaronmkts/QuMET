from typing import Dict
from pennylane.qnn import TorchLayer as TorchConnector
import numpy as np
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
seed = 89
np.random.seed(seed)
torch.manual_seed(seed)

class MMD:

    def __init__(self, scales, space):
        gammas = 1 / (2 * (scales ** 2))
        sq_dists = np.abs(space[:, None] - space[None, :]) ** 2
        self.K = sum(np.exp(-gamma * sq_dists) for gamma in gammas) / len(scales)
        self.K = torch.tensor(self.K, dtype=torch.float64)
        self.scales = scales

    def k_expval(self, px, py):
        return torch.matmul(px, torch.matmul(self.K, py))

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)


bandwidth = np.array([0.25])
space = np.arange(2**9)
mmd = MMD(bandwidth, space)

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
        
        self.q_device = qml.device(self.device, wires=self.n_qubits)
        self._construct_quantum_layer()

    def _construct_quantum_layer(self):
        qnode = qml.QNode(self.circuit, self.q_device, interface="torch", diff_method=self.diff_method)
        weight_shapes = {"weights": (self.depth, self.n_qubits, 3)}

        self.q_layer = TorchConnector(qnode, weight_shapes)

    def circuit(self, inputs, weights):

        qml.StronglyEntanglingLayers(
            weights=weights, ranges=[1] * self.depth, wires=range(self.n_qubits)
        )

        return qml.probs()


    def forward(self, x):
        dummy_input = torch.tensor([])
        prob_distribution = self.q_layer(dummy_input)
        mmd_loss = mmd(prob_distribution, x)
        return mmd_loss, prob_distribution

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

    def forward(self, x):
        return self.generator(x)


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
