import torch
import torch.nn as nn
from torchmetrics import KLDivergence
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import collections

from ..base import WrapperBase


class QGANGenerationModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info,
        learning_rate=1e-4,
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
        self.entropy_val = KLDivergence()
        
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
        self.log("train_g_loss_step", errG, prog_bar=True)
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
        self.log("train_d_loss_step", errD, prog_bar=True)
        self.manual_backward(errD)
        optD.step()
        self.untoggle_optimizer(optD)

    def validation_step(self, batch) -> None:
        # data and real/fake labels
        real_data, coords = batch
        real_data = real_data.view(1, -1)

        input = torch.tensor([])
        fake_data = (
            self.model.generator(input).type_as(real_data).unsqueeze(0)
        )  # should be using model.forward
        self.countor_map = self.get_image(real_data.cpu().numpy(), fake_data.cpu().numpy())
        self.projection = self.image(fake_data.cpu())
        self.entropy_val(fake_data, real_data)
        self.modes, self.high_quality_ratio = self.modes_captured(fake_data)

    def on_validation_epoch_end(self) -> None:
        self.log("val_kl_epoch", self.entropy_val, prog_bar=True)
        self.log('modes_captured', self.modes)
        self.log('high_quality_ratio', self.high_quality_ratio)
        self.logger.experiment.add_image("2D_Countour_Map", self.plot_to_image(self.countor_map), 
                                         global_step=self.current_epoch)
        self.logger.experiment.add_image("Generator_Projection", self.plot_to_image(self.projection), 
                                         global_step=self.current_epoch)
        
    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        match self.optimizer:
            case "adam":
                b1 = 0.777
                b2 = 0.999

                optG = torch.optim.Adam(
                    self.model.generator.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optD = torch.optim.Adam(
                    self.model.discriminator.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")

        return [optG, optD], []
    
    def modes_captured(self, probs_data):
        num_discrete_values = 2 ** (self.model.generator.n_qubits // 2)
        coords = np.linspace(-3, 3, num_discrete_values)
        grid_elements = np.transpose(
            [
                np.tile(coords, len(coords)),
                np.repeat(coords, len(coords)),
            ]
        )
        num_samples = len(grid_elements)
        index_list = list(range(num_samples))
        sampled_integers = np.random.choice(
            index_list, size=self.model.generator.shots, p=probs_data.squeeze().cpu().numpy()
        )
        sampled_coords = grid_elements[sampled_integers]
        
        radius = 2
        thetas = np.linspace(0, 2 * np.pi, 8)
        xs, ys = radius * np.sin(thetas), radius* np.cos(thetas)
        MEANS = np.stack([xs, ys]).transpose()
        STD = 0.1

        l2_store = []
        for x_ in sampled_coords:
            l2_store.append([np.sum((x_ - i) ** 2) for i in MEANS])
        mode = np.argmin(l2_store, 1).flatten().tolist()
        dis_ = [l2_store[j][i] for j, i in enumerate(mode)]
        mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i]) <= (3 * STD)]
        high_quality_ratio = sum(collections.Counter(mode_counter).values()) / float(self.model.generator.shots)
        modes = len(collections.Counter(mode_counter))
        
        return modes, high_quality_ratio

    def image(self, prob_data):

        num_discrete_values = 2 ** (self.model.generator.n_qubits // 2)
        coords = np.linspace(-3, 3, num_discrete_values)
        mesh_x, mesh_y = np.meshgrid(coords, coords)
        grid_shape = (num_discrete_values, num_discrete_values)

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(prob_data, grid_shape)
        surf = ax.plot_surface(mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

    
    def get_image(self, real_data, fake_data):

        def contour_figure(data, cmap, alpha):
 
            num_discrete_values = 2 ** (self.model.generator.n_qubits// 2)
            coords = np.linspace(-3, 3, num_discrete_values)
            mesh_x, mesh_y = np.meshgrid(coords, coords)
            grid_shape = (num_discrete_values, num_discrete_values)
          
            prob_grid = np.reshape(data, grid_shape)

            plt.contourf(mesh_x, mesh_y, prob_grid, cmap=cmap, antialiased=False, alpha = alpha)

        cmap = [cm.Reds, cm.Blues]
        contour_figure(real_data, cmap[0],0.85)
        contour_figure(fake_data, cmap[1], 0.55)