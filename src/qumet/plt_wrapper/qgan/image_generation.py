import torch
import torch.nn as nn
import numpy as np
import math
from ..base import WrapperBase, ValidationResult
from abc import abstractmethod
from line_profiler import profile
import torch.nn.functional as F
from ...dataset.vision.transforms import DataTransformationMixin 
import itertools
import torch.distributions as D
import torch.nn.functional as F

class QGANImageGenerationModelWrapper(WrapperBase, DataTransformationMixin):
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
        self.validation_step_outputs = []
        # Noise utils
        self.validation_z = self.generate_noise('uniform-angle', batch_size = 16)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
        
    @abstractmethod
    def training_step(self, batch):
        pass
        
    def validation_step(self, batch, batch_idx):
        img, _ = batch
        noise = torch.rand(img.size(0), self.n_qubits)
        fake_imgs = self.model(noise)

        return ValidationResult(real_image=img, fake_image=fake_imgs)

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

        self.validation_z = self.generate_noise('uniform-angle', 16)

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
        fake_data = self.model(noise).type_as(real_data)

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

class PQWGANWrapper(QGANImageGenerationModelWrapper):
    def __init__(self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=100,
        optimizer=None):
        super().__init__(model, dataset_info, learning_rate, 
                         weight_decay, epochs, optimizer)

    def training_step(self, batch):
        
        optG, optD = self.optimizers()
       
        # data and real/fake labels
        train_data, _ = batch
        real_data = train_data.reshape(-1, 28, 28)

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
        breakpoint()
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

        # reconstruction weight in discriminator feature space, first tune this parameter if performace is unsatifactory.
        self.recon_weight =  1e-4
    
    def normal_kld(mu, log_sigma):
        kl_divergence = -0.5 * torch.sum(1 + 2 * log_sigma - mu ** 2 - torch.exp(2 * log_sigma), dim=-1).mean(dim=0)
        return kl_divergence
    
    def training_step(self, batch):
        optAE, optD = self.optimizers()
        breakpoint()
        # Data and real/fake labels

        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float).type_as(
            real_imgs
        )
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float).type_as(
            real_imgs
        )

        # Training the autoencoder and generator

        self.toggle_optimizer(optAE)
        optAE.zero_grad()

        mu, log_var, z, recon_imgs = self.model.vae_forward(real_imgs)

        outD_recon, recon_features = self.model.discriminate(recon_imgs)

        prior_loss = self.normal_kld(mu, log_var)
        recon_loss = 6

        # NOTE: this paper says also use recon samples as , 
        # but the official code doesn't use recon images as negative samples

        errG = self.criterion(outD_recon, real_labels)

        self.manual_backward(prior_loss+recon_loss, retain_graph=True)

        for p in self.model.generator.parameters():
            p.grad *= self.recon_weight

        # encoder is not optimized w.r.t. GAN loss

        self.manual_backward(errG, inputs=list(self.model.generator.parameters()), retain_graph=True)
        optAE.step()

        self.untoggle_optimizer(optAE)

        # Training the discriminator

        self.toggle_optimizer(optD)
        optD.zero_grad()

        outD_real = self.model.discriminator(real_imgs).view(-1)
        outD_recon = self.model.discriminator(recon_imgs.detach()).view(-1) #Do i detach here?
        errD_real = self.criterion(outD_real, real_labels)  # Discriminator real loss
        errD_recon = self.criterion(outD_recon, fake_labels)  # Discriminator fake loss

        errD = (errD_real + errD_recon) 

        self.manual_backward(errD, inputs=list(self.model.discriminator.parameters()))
        optD.step()

        self.untoggle_optimizer(optD)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        N = imgs.size(0)
        mu, log_var, z, recon_imgs = self.model.vae_forward(imgs) 
        val_mse = F.mse_loss(imgs, recon_imgs)
        self.log("val_log/van_mse", val_mse)

        return ValidationResult(real_image=imgs, fake_image=recon_imgs, 
                    recon_image=recon_imgs, label=labels, encode_latent=z)
  
    def configure_optimizers(self):
        lrG = 0.3  # Learning rate for the generator
        lrD = 0.01  # Learning rate for the discriminator
    
        match self.optimizer.lower():
            case "adam":

                b1 = 0.777
                b2 = 0.999

                optAE = torch.optim.Adam(
                    itertools.chain(self.model.encoder.parameters(), 
                    self.model.generator.parameters()),
                    lr=lrG,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optD = torch.optim.Adam(
                    self.model.discriminator.parameters(),
                    lr=lrD,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )
            case "sgd":

                optAE = torch.optim.SGD(
                    itertools.chain(self.model.encoder.parameters(), 
                    self.model.generator.parameters()),
                    lr = lrG)
                
                optD = torch.optim.SGD(
                    self.model.discriminator.parameters(), 
                    lr=lrD)
                
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")
            
        return [optAE, optD], []

        

       