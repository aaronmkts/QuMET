import torch
import torch.nn as nn
from torchmetrics import KLDivergence
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from ..base import WrapperBase
from scipy.stats import entropy

class QGANDiscreteGenModelWrapper(WrapperBase):
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

    def forward(self, batch, measure):
        return self.model(batch, measure)

    
    def training_step(self, batch, batch_idx):

      
        mode = 'train'
        optG, optD = self.optimizers()

        # Set sata and real/fake labels
        real_data = batch
        
        real_labels = torch.full((real_data.size(0),), 1.0, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((real_data.size(0),), 0.0, dtype=torch.float).type_as(
            real_data
        )

        batch_size = real_data.size(0)

        self.toggle_optimizer(optD)
        optD.zero_grad()

        discriminator_training_steps = 5
        for _ in range(discriminator_training_steps):
            fake_data = self.model.generator(batch_size, mode).type_as(real_data)

            outD_real = self.model.discriminator(real_data)
            outD_fake = self.model.discriminator(fake_data.detach()) 
   
            errD_real = self.criterion(outD_real, real_labels) # Discriminator real loss
            errD_fake = self.criterion(outD_fake, fake_labels) # Discriminator fake loss
            errD = (errD_real + errD_fake) 
            
            self.manual_backward(errD)
            optD.step()
            self.log("Discriminator_loss", errD, prog_bar=True)

        self.untoggle_optimizer(optD)


        self.toggle_optimizer(optG)
        # Training the generator
    
        optG.zero_grad()
        outD_fake = self.model.discriminator(fake_data).detach() #do i detach here?
        errG = self.criterion(outD_fake, real_labels) 

        gradients = self.model.generator.get_gradient(batch_size, self.model.discriminator)
        for param in self.model.generator.parameters():
            param.grad = gradients
            param.backward(gradients)
    
        optG.step()
        self.log("Generator_loss", errG, prog_bar=True)
        self.untoggle_optimizer(optG)

    def validation_step(self, batch) -> None:
        # data and real/fake labels
        mode = 'evaluate'
        real_data = batch[0].unsqueeze(0) # all elements of the batch compose of the original prob_data
        
        sample_size = 50000
        fake_data = (
            self.model.generator(sample_size, mode).type_as(real_data).unsqueeze(0)
        )  
   
       # self.countor_map = self.contour_plot(real_data.cpu().numpy(), fake_data.cpu().numpy())
        epsilon = 1e-6
    
        self.entropy_val(fake_data + epsilon, real_data + epsilon)
        self.gen_out = self.density_plot(fake_data.cpu())
        


    def on_validation_epoch_end(self) -> None:
        self.log("val_kl_epoch", self.entropy_val, prog_bar = True, on_epoch = True)
        '''     
        self.logger.experiment.add_image("2D_Countour_Map", self.plot_to_image(self.countor_map), 
                                         global_step=self.current_epoch)
        self.logger.experiment.add_image("Generator_Projection", self.plot_to_image(self.gen_out), 
                                         global_step=self.current_epoch)
        '''
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
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")

        return [optG, optD], []
    
    def density_plot(self, prob_data):
        num_discrete_values = 2 ** (self.n_qubits // 2)
        coords = np.linspace(-3, 3, num_discrete_values)
        mesh_x, mesh_y = np.meshgrid(coords, coords)
        grid_shape = (num_discrete_values, num_discrete_values)

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(prob_data, grid_shape)
        surf = ax.plot_surface(mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig
'''
    def contour_plot(self, real_data, fake_data):

        def contour_figure(data, cmap, alpha):
            num_discrete_values = 2 ** (self.n_qubits// 2)
            coords = np.linspace(-3, 3, num_discrete_values)
            mesh_x, mesh_y = np.meshgrid(coords, coords)
            grid_shape = (num_discrete_values, num_discrete_values)
          
            prob_grid = np.reshape(data, grid_shape)

            plt.contourf(mesh_x, mesh_y, prob_grid, cmap=cmap, antialiased=False, alpha = alpha)

        cmap = [cm.Reds, cm.Blues]
        contour_figure(real_data, cmap[0],0.85)
        contour_figure(fake_data, cmap[1], 0.55)

        '''