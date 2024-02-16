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

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as L

_CONFIG_FOR_DOC = "QGCD_Binary_Config"

pi = math.pi

class Binary_Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, config: QGCD_Binary_Config):
        super(Binary_Discriminator,self).__init__()
        self.input_size = config.input_size

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
            #Third hidden layer (64 -> num_output_features)
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        #self.model.apply(self.init_weights)
    
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight)
            torch.nn.init.kaiming_uniform_(layer.bias)

    def forward(self, 
                input):
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
        self.device = qml.device(config.device, wires = config.n_qubits, shots = config.shots)
        q_weight_shapes = {"q_weights_y": (self.depth, self.n_qubits),
                           "q_weights_z": (self.depth, self.n_qubits)}
        init_method = {"q_weights_y": lambda x : torch.nn.init.uniform_(x, -pi, pi),
                        "q_weights_z": lambda x : torch.nn.init.uniform_(x, -pi, pi)}
        q_generator = qml.QNode(self._circuit, self.device, interface="torch")
        self.q_generator = TorchConnector(q_generator, q_weight_shapes, init_method = init_method)

    def __str__(self):
        return f"QuantumGenerator({self.n_qubits}) "


    def _circuit(self, inputs, q_weights_y, q_weights_z):
        """Builds the circuit to be fed to the connector as a QML node"""
        
        # Repeated layer
        for i in range(self.depth):
            for y in range(self.n_qubits):
                qml.RY(q_weights_y[i][y], wires = y)
                qml.RZ(q_weights_z[i][y], wires = y)
            for y in range(self.n_qubits - 1):
                qml.CNOT(wires=[y, y + 1])

        return qml.probs() 

    def forward(self, inputs: Tensor):
        return self.q_generator(inputs)


class GAN(L.LightningModule):
    def __init__(
        self,
        config = QGCD_Binary_Config,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        
        self.generator = Binary_Generator(config = config)
        self.discriminator = Binary_Discriminator(config)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        real_data = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn([])
        z.type_as(real_data)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_data = self(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(real_data.size(), 1)
        valid = valid.type_as(real_data)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(real_data.size(), 1)
        valid = valid.type_as(real_data)

        real_loss = self.adversarial_loss(self.discriminator(real_data), valid)

        # how well can it label as fake?
        fake = torch.zeros(real_data.size(), 1)
        fake = fake.type_as(real_data)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr = 4e-4
        b1 = 0.777
        b2 = 0.999

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []