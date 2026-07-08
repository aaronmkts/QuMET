"""Quantum implicit neural representation GAN model."""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from logging import getLogger
from pennylane.qnn import TorchLayer as TorchConnector
from typing import Any, Callable, Dict, List, Optional, Type, Union
from torch import Tensor
from qw_map import arctan
logger = getLogger(__name__)
pi = math.pi


image_shape = (1, 28, 28)
# fmt:0ff
config = {
    "discriminator": {"image_shape": image_shape},
    "generator": {
        'image_shape': image_shape,
        'in_features': 6,
        'hidden_features': 6,
        'hidden_layers': 2,
        'out_features': 1,
        'spectrum_layer': 2,
        'use_noise': 0,
        'outermost_linear': True,
    },
}
# fmt:on
class Discriminator(nn.Module):
    """Classical image discriminator for QINR.

    Args:
        config: Nested QINR configuration containing image shape settings.
        task: QuMET task name used by the caller.
    """

    def __init__(self, config, task):
        super().__init__()
        name = "discriminator"
        self.image_shape = config[name]["image_shape"]

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

        # Apply LeCun initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                # LeCun initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x
    
class QuantumLayer(nn.Module):
    """Quantum layer used inside the QINR hybrid generator.

    Args:
        in_features: Number of input features and circuit wires.
        spectrum_layer: Number of strongly entangling layer blocks.
        use_noise: Optional random rotation scale for stochastic generation.
    """

    def __init__(self, in_features, spectrum_layer, use_noise):
        super().__init__()

        self.in_features = in_features
        self.n_layer = spectrum_layer
        self.use_noise = use_noise

        def _circuit(inputs, weights1, weights2):
            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)
                for j in range(self.in_features):
                    qml.RZ(inputs[j], wires=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angle = np.pi + self.use_noise * np.random.rand()
                    qml.RX(rand_angle, wires=i)

            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            return res

        torch_device = qml.device('default.qubit', wires=in_features)
        weight_shape = {"weights1": (self.n_layer, 2, in_features, 3), "weights2": (2, in_features, 3)}

        self.qnode = qml.QNode(_circuit, torch_device, diff_method="backprop", interface="torch")

        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        orgin_shape = list(x.shape[0:-1]) + [-1]
        if len(orgin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        outputs = []
        for item in x:
            out = self.qnn(item)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=0)
        return outputs.reshape(orgin_shape)


class HybridLayer(nn.Module):
    """Linear projection followed by a quantum layer.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension before the quantum layer.
        spectrum_layer: Number of strongly entangling layer blocks.
        use_noise: Optional random rotation scale for stochastic generation.
        bias: Whether the linear projection uses a bias term.
        idx: Layer index used for diagnostics.
    """

    def __init__(self, in_features, out_features, spectrum_layer, use_noise, bias=True, idx=0):
        super().__init__()
        self.idx = idx
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        self.qlayer = QuantumLayer(out_features, spectrum_layer, use_noise)
        
    def forward(self, x):
        x1 = self.clayer(x)
        out = self.qlayer(x1)
        return out
    

class Generator(nn.Module):
    """Hybrid generator for QINR image synthesis.

    Args:
        config: Nested QINR configuration containing generator settings.
        task: QuMET task name used by the caller.
    """

    def __init__(self, config, task):
        name = "generator"
        image_shape = config[name]["image_shape"]
        in_features = config[name]["in_features"]
        hidden_features = config[name]["hidden_features"]
        hidden_layers = config[name]["hidden_layers"]
        out_features = config[name]["out_features"]
        spectrum_layer = config[name]["spectrum_layer"]
        use_noise = config[name]["use_noise"]
        outermost_linear = config[name]["outermost_linear"]
        self.n_qubits = config[name]["hidden_features"]


        super().__init__()

        self.net = []
        self.net.append(HybridLayer(in_features, hidden_features, spectrum_layer, use_noise, idx=1))

        for i in range(hidden_layers):
            self.net.append(HybridLayer(hidden_features, hidden_features, spectrum_layer, use_noise, idx=i + 2))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, 128)

        else:
            final_linear = HybridLayer(hidden_features, out_features, spectrum_layer, use_noise)
       
        final_linear_1 = nn.Linear(128, 512)
        final_linear_2 = nn.Linear(512, 256)
        final_linear_3 = nn.Linear(256, int(np.prod(image_shape)))
        self.net.append(final_linear)
        self.net.append(final_linear_1)
        self.net.append(final_linear_2)
        self.net.append(final_linear_3)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):

        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        final_out_new = output.view(output.shape[0], *image_shape)
        
        return final_out_new
    

class QINR(nn.Module):
    """QINR GAN model with hybrid generator and classical discriminator.

    Args:
        config: Nested QINR configuration for discriminator and generator.
        task: QuMET task name used by the caller.
    """

    def __init__(self, config, task):
        super().__init__()
        # networks
      
        self.discriminator = Discriminator(config, task)
        self.generator = Generator(config, task)

    def forward(self, z: Tensor):
        with torch.no_grad():
            imgs_batch = self.generator(z)
            imgs = imgs_batch.view(-1, 1, 28, 28)
        return imgs
    
# ---------------------------------------
# QINR QGAN
# ---------------------------------------

def _qinr_qc(config, task: str) -> QINR:

    model = QINR(config, task)
    return model


def get_qinr_qc(info: Dict) -> QINR:
    """Build a QINR model from dataset metadata.

    Args:
        info: Dataset metadata passed by the QuMET model registry.

    Returns:
        Configured QINR instance.
    """

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into QINR QGAN")
    return _qinr_qc(config=config, task=task)
