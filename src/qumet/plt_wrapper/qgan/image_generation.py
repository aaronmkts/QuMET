import torch
import torch.nn as nn
from torchmetrics import KLDivergence
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math
import collections
from scipy.linalg import sqrtm
from ..base import WrapperBase
from abc import abstractmethod
from line_profiler import profile
import torch.nn.functional as F

class QGANImageGenerationModelWrapper(WrapperBase):
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
        self.image_size = dataset_info.image_size[1:]
        self.dataset_info = dataset_info
        self.optimizer = optimizer
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_qubits = self.model.generator.n_qubits
        self.criterion = nn.BCELoss()
        # Noise utils
        self.fixed_noise = self.generate_noise('uniform-angle', batch_size = 8)

        
    @abstractmethod
    def training_step(self, batch):
        pass
    
        '''

    def validation_step(self, batch) -> None:
        #calc fid
        #calc jsd div 
        #calc is
        #calc mode score

    def on_validation_epoch_end(self) -> None:
        self.log("FID")
        self.log('IS')
        self.log('JSD')
        self.logger.experiment.add_image("2D_Countour_Map")
        '''
    
    #NOISE FUNCTIONS

    def relu(self, x):
        return x * (x > 0)
    
    def get_noise_upper_bound(self, gen_loss, disc_loss, original_ratio):
        R = disc_loss.detach().numpy()/gen_loss.detach().numpy()
        return math.pi/8 + (5 *math.pi / 8) * self.relu(np.tanh((R - (original_ratio))))

    def generate_noise(self, noise_type, batch_size=8):
        match noise_type:
            case 'uniform-angle':
                return torch.rand(batch_size, self.n_qubits) * math.pi / 2
            case 'uniform':
                return torch.rand(batch_size, self.n_qubits)
            case 'gaussian':
                return torch.randn(batch_size, self.n_qubits)
            case 'adaptive':
                # Define your custom noise generation logic here
                return torch.rand(batch_size, self.n_qubits) * self.noise_upper_bound
            case _:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        match self.optimizer.lower():
            case "adam":
                b1 = 0.777
                b2 = 0.999

                optG = torch.optim.Adam(
                    self.model.generator.parameters(),
                    lr=0.03,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optD = torch.optim.Adam(
                    self.model.discriminator.parameters(),
                    lr=0.01,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )
            case "sgd":
                lrG = 0.3  # Learning rate for the generator
                lrD = 0.01  # Learning rate for the discriminator

                optD = torch.optim.SGD(
                    self.model.discriminator.parameters(), 
                    lr=lrD)
                optG = torch.optim.SGD(self.model.generator.parameters(),
                                        lr=lrG)
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")

        return [optG, optD], []
    
class PatchGANWrapper(QGANImageGenerationModelWrapper):
    def __init__(self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=100,
        optimizer=None):
        super().__init__(model, dataset_info, learning_rate, 
                         weight_decay, epochs, optimizer)

        self.fixed_noise = self.generate_noise('uniform-angle', 16)

    
    def training_step(self, batch):
        optG, optD = self.optimizers()
        
        # data and real/fake labels
        train_data, _ = batch
        real_data = train_data.reshape(-1, self.image_size[0] * self.image_size[1])

        batch_size = real_data.size(0)

        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float).type_as(
            real_data
        )
        
        # Generate fake-data using noise input
        noise = self.generate_noise('uniform-angle', batch_size)

        fake_data = self.model.generator(noise).type_as(real_data)

        # Training the discriminator
        self.toggle_optimizer(optD)

        optD.zero_grad()
        
        outD_real = self.model.discriminator(real_data).view(-1)
        outD_fake = self.model.discriminator(fake_data.detach()).view(-1)
        errD_real = self.criterion(outD_real, real_labels)  # Discriminator real loss
        errD_fake = self.criterion(outD_fake, fake_labels)  # Discriminator fake loss
        
        self.manual_backward(errD_real)
        self.manual_backward(errD_fake)

        errD = (errD_real + errD_fake) 
        self.log("train_d_loss_step", errD, prog_bar=True)
        optD.step()

        self.untoggle_optimizer(optD)
        
        # Training the generator
        self.toggle_optimizer(optG)
        optG.zero_grad()
        outD_fake = self.model.discriminator(fake_data).view(-1)
        errG = self.criterion(outD_fake, real_labels)
        
        self.manual_backward(errG)

        self.log("train_g_loss_step", errG, prog_bar=True)
        optG.step()

        self.untoggle_optimizer(optG)

class MosaiQGANWrapper(QGANImageGenerationModelWrapper):
    def __init__(self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=100,
        optimizer=None):
        super().__init__(model, dataset_info, learning_rate, 
                         weight_decay, epochs, optimizer)

        self.pca_dims = 40
        self.original_ratio = None
        self.upper_bounds = [math.pi / 8]
        self.noise_upper_bound = math.pi / 8

    def training_step(self, batch):
        
        optG, optD = self.optimizers()
        
        # data and real/fake labels

        pca_data, _ = batch
        batch_size = pca_data.size(0)
        real_data = pca_data.reshape(-1, self.pca_dims)

    
        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float).type_as(
            real_data
        )
        
        # Generate fake-data using noise input
        noise = self.generate_noise('adaptive', batch_size)
        fake_data = self.model.generator(noise, batch_size).type_as(real_data)

        # Training the discriminator
        self.toggle_optimizer(optD)

        optD.zero_grad()
       
        outD_real = self.model.discriminator(real_data).view(-1)
        outD_fake = self.model.discriminator(fake_data.detach()).view(-1)
        errD_real = self.criterion(outD_real, real_labels)  # Discriminator real loss
        errD_fake = self.criterion(outD_fake, fake_labels)  # Discriminator fake loss
        
        self.manual_backward(errD_real)
        self.manual_backward(errD_fake)

        errD = (errD_real + errD_fake) 
        self.log("train_d_loss_step", errD, prog_bar=True)
        optD.step()

        self.untoggle_optimizer(optD)
        
        # Training the generator
        self.toggle_optimizer(optG)

        optG.zero_grad()
        outD_fake = self.model.discriminator(fake_data).view(-1)
        errG = self.criterion(outD_fake, real_labels)
        self.manual_backward(errG) 
        optG.step()

        if self.original_ratio is None:
                self.original_ratio = errD.detach().numpy()/errG.detach().numpy()
        noise_upper_bound = self.get_noise_upper_bound(errG, errD, self.original_ratio)
        self.upper_bounds.append(noise_upper_bound)
        self.log("train_g_loss_step", errG, prog_bar=True)
    
        self.untoggle_optimizer(optG)

class SSPQGANWrapper(QGANImageGenerationModelWrapper):
    def __init__(self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=100,
        optimizer=None):
        super().__init__(model, dataset_info, learning_rate, 
                         weight_decay, epochs, optimizer)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
        
    def vae_loss(self, real_data, fake_data, mu, log_var):
        # Reconstruction loss
        recon_loss = self.criterion(fake_data, real_data)
        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div
    
    def discriminator_loss(self, real_labels, fake_labels, real_data, fake_data):
    
        outD_real = self.model.discriminator(real_data).view(-1)
        outD_fake = self.model.discriminator(fake_data.detach()).view(-1)
        errD_real = self.criterion(outD_real, real_labels)  # Discriminator real loss
        errD_fake = self.criterion(outD_fake, fake_labels)

        return errD_real, errD_fake
    
    def generator_loss(self, real_labels, fake_data, real_data):
        #Gan loss + reconstruction loss
        outD_fake = self.model.discriminator(fake_data).view(-1)
        errG = self.criterion(outD_fake, real_labels)

        recon_loss = F.binary_cross_entropy(fake_data, real_data, reduction='sum')
        return errG + recon_loss

    
    def training_step(self, batch):
        optG, optD, optVAE = self.optimizers()
        
        # data and real/fake labels
        train_data, _ = batch
        real_data = train_data.reshape(-1, self.image_size[0] * self.image_size[1])

        batch_size = real_data.size(0)

        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float).type_as(
            real_data
        )

        mu, log_var, fake_data, z = self.model.forward(real_data)

        # Training the VAE
        self.toggle_optimizer(optVAE)
        optVAE.zero_grad()

        errVAE = self.vae_loss(real_data, fake_data, mu, log_var)

        self.manual_backward(errVAE)
        optVAE.step()
        self.untoggle_optimizer(optVAE)

        # Training the Generator
        self.toggle_optimizer(optG)
        optG.zero_grad()

        errG = self.generator_loss(real_labels, fake_data, real_data)
        self.manual_backward(errG)
        optG.step()
        self.untoggle_optimizer(optG)

        # Training the Discriminator
        self.toggle_optimizer(optD)
        optD.zero_grad()

        errD_real, errD_fake = self.discriminator_loss(real_labels, fake_labels, real_data, fake_data)
        errD = (errD_real + errD_fake)
        self.manual_backward(errD)
        optD.step()
        self.untoggle_optimizer(optD)

         # Log the losses
        self.log("train_vae_loss_step", errVAE, prog_bar=True)
        self.log("train_g_loss_step", errG, prog_bar=True)
        self.log("train_d_loss_step", errD, prog_bar=True)
  
    def configure_optimizers(self):

    # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        match self.optimizer.lower():
            case "adam":
                b1 = 0.777
                b2 = 0.999

                optG = torch.optim.Adam(
                    self.model.generator.parameters(),
                    lr=0.03,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optD = torch.optim.Adam(
                    self.model.discriminator.parameters(),
                    lr=0.01,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optVAE = torch.optim.Adam(
                    self.model.encoder.parameters(),
                    lr=0.01,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )
            case "sgd":
                lrG = 0.3  # Learning rate for the generator
                lrD = 0.01  # Learning rate for the discriminator

                optD = torch.optim.SGD(
                    self.model.discriminator.parameters(), 
                    lr=lrD)
                optG = torch.optim.SGD(self.model.generator.parameters(),
                                        lr=lrG)
                
                optVAE = torch.optim.SGD(self.model.encoder.parameters(),
                                         lr = 0.01)
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")
            
        return [optG, optD, optVAE], []

        

       