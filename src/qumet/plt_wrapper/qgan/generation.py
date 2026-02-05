import torch

from ..base import WrapperBase


class QGANGenerationModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=100,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.optimizer = optimizer
        self.automatic_optimization = False

    def adversarial_loss(self, input, target, w):
        bce_loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
        weighted_loss = w * bce_loss
        total_loss = -torch.sum(weighted_loss)
        return total_loss

    def training_step(self, batch):

        optG, optD = self.optimizers()

        # data and real/fake labels
        real_data, coords = batch

        real_labels = torch.full((real_data.size(0),), 1.0, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((real_data.size(0),), 0.0, dtype=torch.float).type_as(
            real_data
        )

        # Generate fake-data using noise input
        input = torch.tensor([])
        fake_data = self.model.generator(input).type_as(real_data).reshape(-1, 1)

        # Configure Samples
        disc_value = self.model.discriminator(coords)

        # Training the generator
        self.toggle_optimizer(optG)
        optG.zero_grad()
        errG = self.adversarial_loss(disc_value, real_labels, fake_data)
        self.log("d_loss", errG, prog_bar=True)
        self.manual_backward(errG, retain_graph=True)
        optG.step()
        self.untoggle_optimizer(optG)

        # Training the discriminator
        self.toggle_optimizer(optD)
        optD.zero_grad()
        errD_real = self.adversarial_loss(
            disc_value, real_labels, real_data
        )  # Discriminator real loss
        errD_fake = self.adversarial_loss(
            disc_value, fake_labels, fake_data.detach()
        )  # Discriminator fake loss
        errD = (errD_real + errD_fake) / 2
        self.log("d_loss", errD, prog_bar=True)
        self.manual_backward(errD)
        optD.step()
        self.untoggle_optimizer(optD)

    def configure_optimizers(self):

        lr = 4e-4
        b1 = 0.777
        b2 = 0.999

        opt_g = torch.optim.Adam(
            self.model.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        return [opt_g, opt_d], []
