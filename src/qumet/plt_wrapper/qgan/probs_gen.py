import torch
import torch.nn as nn
from torchmetrics import KLDivergence

from ..base import WrapperBase


class QGANProbsGenModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info,
        learning_rate=1e-2,
        weight_decay=0.0,
        epochs=100,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            optimizer=optimizer,
        )

        self.optimizer = optimizer
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_qubits = self.model.generator.n_qubits
        self.entropy_val = KLDivergence()
        self.criterion = nn.BCELoss()

    def training_step(self, batch, batch_idx):

        optG, optD = self.optimizers()

        # Set sata and real/fake labels
        real_data = batch.reshape(
            -1,
        )

        real_labels = torch.full((batch.size(0),), 1.0, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((batch.size(0),), 0.0, dtype=torch.float).type_as(
            real_data
        )

        self.toggle_optimizer(optD)
        optD.zero_grad()

        discriminator_training_steps = 1
        for _ in range(discriminator_training_steps):
            fake_data = self.model.generator().type_as(real_data)
            outD_real = self.model.discriminator(real_data)
            outD_fake = self.model.discriminator(fake_data.detach())

            errD_real = self.criterion(
                outD_real, real_labels
            )  # Discriminator real loss
            errD_fake = self.criterion(
                outD_fake, fake_labels
            )  # Discriminator fake loss
            errD = errD_real + errD_fake

            self.manual_backward(errD)
            optD.step()
            self.log("Discriminator_loss", errD, prog_bar=True)

        self.untoggle_optimizer(optD)

        self.toggle_optimizer(optG)
        # Training the generator
        fake_data = self.model.generator().type_as(real_data)

        optG.zero_grad()
        outD_fake = self.model.discriminator(fake_data)  # do i detach here?
        errG = self.criterion(outD_fake, real_labels)

        self.manual_backward(errG)
        optG.step()
        self.log("Generator_loss", errG, prog_bar=True)
        self.untoggle_optimizer(optG)

        epsilon = 1e-6
        kl_div = self.entropy_val(
            fake_data.unsqueeze(0) + epsilon, real_data.unsqueeze(0) + epsilon
        ).detach()
        self.log("kl_div", kl_div, prog_bar=True)

    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        match self.optimizer:
            case "adam":
                b1 = 0.7
                b2 = 0.999

                optG = torch.optim.Adam(
                    self.model.generator.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optD = torch.optim.Adam(
                    self.model.discriminator.parameters(),
                    lr=0.0001,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")

        return [optG, optD], []
