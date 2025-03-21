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
        "latent_dim": 7,
        'image_shape': image_shape,
    },
}
# fmt:on

#DISCRIMINATOR

class Discriminator(nn.Module):
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

class Generator(nn.Module):
    def __init__(self, config, task):
        super(Generator, self).__init__()
        name = "generator"
        self.latent_dim = config[name]["latent_dim"]
        self.image_shape = config[name]["image_shape"]
        self.output_dim = int(np.prod(self.image_shape)) 

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        # Apply LeCun initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                # LeCun initialization for layers with LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.model(z)
        x = x.view(z.size(0), *self.image_shape)
        return x

class GAN(nn.Module):
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
# ClassicalGan
# ---------------------------------------

def _gan(config, task: str) -> GAN:

    model = GAN(config, task)
    return model


def get_gan(info: Dict) -> GAN:

    task = "info.generation"
    logger.info(f"The following {config} loaded for task into PatchQuantumGenerator")
    return _gan(config=config, task=task)