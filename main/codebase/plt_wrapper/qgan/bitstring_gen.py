import torch
import torch.nn as nn
from torchmetrics import KLDivergence
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from ..base import WrapperBase
from scipy.stats import entropy

class QGANBitstringGenerationModelWrapper(WrapperBase):
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
        measure = 0
        optG, optD = self.optimizers()

        # data and real/fake labels
        real_data = batch
        
        real_labels = torch.full((real_data.size(0),), 1.0, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((real_data.size(0),), 0.0, dtype=torch.float).type_as(
            real_data
        )

        batch_size = real_data.size(0)


        # Generate fake-data
        fake_data = self.model.generator(batch_size, measure).type_as(real_data)


        self.toggle_optimizer(optG)
        # Training the generator
        if (batch_idx ==0) or batch_idx % 5 == 0:
            optG.zero_grad()
            outD_fake = self.model.discriminator(fake_data)
            errG = self.criterion(outD_fake, real_labels) #self.adversarial_loss(outD_fake,real_labels)#self.criterion(outD_fake, real_labels)
            self.manual_backward(errG, retain_graph = True)
            optG.step()
            self.log("train_g_loss_step", errG, prog_bar=True)
    
        self.untoggle_optimizer(optG)

        # Training the discriminator
        self.toggle_optimizer(optD)
        optD.zero_grad()
       
        outD_real = self.model.discriminator(real_data)
        outD_fake = self.model.discriminator(fake_data.detach()) #D_thi(x)

        errD_real = self.criterion(outD_real, real_labels) #self.adversarial_loss(outD_real, real_labels) #self.criterion(outD_real, real_labels)      # Discriminator real loss
        errD_fake = self.criterion(outD_fake, fake_labels)#self.adversarial_loss(outD_fake, fake_labels) #self.criterion(outD_fake, fake_labels)      # Discriminator fake loss]

        errD = (errD_real + errD_fake) 

        self.manual_backward(errD)
        self.log("train_d_loss_step", errD, prog_bar=True)
        optD.step()
        self.untoggle_optimizer(optD)

        


    def validation_step(self, batch) -> None:
        # data and real/fake labels
        measure = 1
        real_data = batch[0].unsqueeze(0) # all elements of the batch compose of the original prob_data
        
        sample_size = 10000
        fake_data = (
            self.model.generator(sample_size, measure).type_as(real_data).unsqueeze(0)
        )  # should be using model.forward, change the shot count here for generato
        
        
        self.gen_out = self.image(fake_data.cpu())
        self.entropy_val(fake_data, real_data)

    def on_validation_epoch_end(self) -> None:
        self.log("val_kl_epoch", self.entropy_val, prog_bar = True, on_epoch = True)
        self.logger.experiment.add_image("gen_output", self.plot_to_image(self.gen_out), 
                                         global_step=self.current_epoch)
        
    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        match self.optimizer:
            case "adam":
                b1 = 0.7
                b2 = 0.999
                
                
                optG = torch.optim.Adam(
                    self.model.generator.parameters,
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
    
    def image(self, prob_data):
        num_discrete_values = 2 ** (self.n_qubits // 2)
        coords = np.linspace(-2, 2, num_discrete_values)
        mesh_x, mesh_y = np.meshgrid(coords, coords)
        grid_shape = (num_discrete_values, num_discrete_values)

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "3d"})
        prob_grid = np.reshape(prob_data, grid_shape)
        surf = ax.plot_surface(mesh_x, mesh_y, prob_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig
    
    def _binary_to_int(self, bit_list):
            output = 0
            for bit in bit_list:
                output = output * 2 + bit
            return int(output)
    
    def get_probs_list(self, bit_list):

        def _binary_to_int(bit_list):
            output = 0
            for bit in bit_list:
                output = output * 2 + bit

            return int(output)
        
        int_list = list(map(_binary_to_int, bit_list))

        count_dict = {integer: 0 for integer in range(2**self.n_qubits)}
        
        # Potentially replace for loop with: count_dict.update(Counter(int_list))
        for integer in int_list:
            count_dict[integer] += 1
        
        total_counts = sum(count_dict.values())
        probs_list = torch.tensor([count / total_counts for count in count_dict.values()]).view(1,-1)

        return probs_list