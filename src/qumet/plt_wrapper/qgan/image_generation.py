import torch
import torch.nn as nn
import numpy as np
import math
from ..base import WrapperBase, ValidationResult
from abc import abstractmethod
import torch.nn.functional as F
import torch.nn.functional as F
from ..utils import compute_gradient_penalty

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
        self.validation_step_outputs = []
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
        noise = torch.randn(img.size(0), self.n_qubits)
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
                b1 = 0
                b2 = 0.9

                optG = torch.optim.Adam(
                    self.model.generator.parameters(),
                    lr=0.01,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optD = torch.optim.Adam(
                    self.model.discriminator.parameters(),
                    lr=0.0005,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )
            case "sgd":
                lrG = 0.3  # Learning rate for the generator
                lrD = 0.001  # Learning rate for the discriminator

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

        real_labels = torch.full((batch_size,), 0.9, dtype=torch.float).type_as(
            real_data
        )
        fake_labels = torch.full((batch_size,), 0.1, dtype=torch.float).type_as(
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

    def validation_step(self, batch, batch_idx):

        transform = self.trainer.datamodule.transform_instance

        pca_data, _ = batch
        batch_size = pca_data.size(0)
        real_imgs = transform.inverse_transform(pca_data)

        noise = self.generate_noise('adaptive', batch_size)
        fake_imgs = transform.inverse_transform(self.model(noise))
       
        return ValidationResult(real_image=real_imgs, fake_image=fake_imgs)

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
        self.lambda_gp = 10
        self.n_critic = 5
        self.validation_z = self.generate_noise('uniform', batch_size = 16)

    def training_step(self, batch, batch_idx):
        
        critic = self.model.discriminator
        optG, optD = self.optimizers()

        # data and real/fake labels
        real_data, _ = batch

        batch_size = real_data.size(0)

        # Generate fake-data using noise input
        noise = self.generate_noise('uniform', batch_size)
        fake_data = self.model.generator(noise).type_as(real_data)

        # Training the discriminator
        self.toggle_optimizer(optD)
        optD.zero_grad()
        
     
        # Real and fake images
        real_validity, fake_validity = critic(real_data), critic(fake_data.detach())
        # # Adversarial loss
        gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data)
        errD = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

        wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)

        self.manual_backward(errD)
        optD.step()

        self.log("discriminator/total_loss", errD, prog_bar=True)
        self.log("wasserstein_distance", wasserstein_distance, prog_bar=True)

        self.untoggle_optimizer(optD)
        
        
        # Training the generator
        if  batch_idx!= 0 and batch_idx % (self.n_critic + 1) == self.n_critic:
            self.toggle_optimizer(optG)
            optG.zero_grad()

            fake_data = self.model.generator(noise).type_as(real_data)
            # Loss measures generator's ability to fool the discriminator,Train on fake images

            fake_validity = critic(fake_data)
            errG = -torch.mean(fake_validity)
        
            self.manual_backward(errG)
            optG.step()
            self.log("generator/total_loss", errG, prog_bar=True)
  
            self.untoggle_optimizer(optG) 

    def validation_step(self, batch, batch_idx):

        img, _ = batch
        N = img.size(0)

        noise = torch.rand(img.size(0), self.n_qubits) 
        fake_imgs = self.model(noise)
        recon_image = self.model(self.validation_z)

        val_mse_sum = F.mse_loss(img, fake_imgs, reduction='sum') / N
        self.log("metrics/val_mse_reduction", val_mse_sum, on_epoch=True)

        return ValidationResult(real_image=img, fake_image=fake_imgs, recon_image=recon_image)

class APQGANWrapper(QGANImageGenerationModelWrapper):
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
        self.recon_weight = 5e-4
        self.lambda_gp = 10
        self.n_critic = 5

    def normal_kld(self, mu, log_var):
        # Compute the variance from the log variance

        prior_loss = 1 + log_var - mu.pow(2) - log_var.exp()
        kl_divergence = torch.mean(-0.5 * torch.sum(prior_loss, dim = 1))

        return kl_divergence

    def training_step(self, batch, batch_idx):

        critic = self.model.discriminator
        optE, optG, optD = self.optimizers()

        # data and real/fake labels
        real_data, _ = batch
       
        batch_size = real_data.size(0)
                
        # Generate fake-data using noise input
        mu, log_var, z = self.model.vae_forward(real_data)
        recon_imgs = self.model(z)

        # Training the discriminator
        self.toggle_optimizer(optD)
        optD.zero_grad()
        
        # Real and fake images
        real_validity  = critic(real_data)
        fake_validity  =  critic(recon_imgs.detach())

        # Adversarial loss
        gradient_penalty = compute_gradient_penalty(critic, real_data, recon_imgs)
        errD = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

        wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)

        self.manual_backward(errD)
        optD.step()

        self.log("discriminator/total_loss", errD, prog_bar=True)
        self.log("wasserstein_distance", wasserstein_distance, prog_bar=True)

        self.untoggle_optimizer(optD)
        # Training the generator

        if batch_idx!= 0 and batch_idx % (self.n_critic) == 0:
        
            # Train the encoder
            self.toggle_optimizer(optE)
            optE.zero_grad()
            
            # Recompute recon_imgs for the encoder
            mu, log_var, z = self.model.vae_forward(real_data)
            recon_imgs = self.model(z)
            
            recon_loss = F.mse_loss(recon_imgs, real_data, reduction='sum') / batch_size
            prior_loss = self.normal_kld(mu, log_var) 

            errE =  prior_loss + recon_loss
            self.manual_backward(errE)
            optE.step()

            self.log('encoder/prior_loss', prior_loss)
            self.log('encoder/recon_loss', recon_loss)
            self.log('encoder/total_loss', errE, prog_bar=True)

            self.untoggle_optimizer(optE)

            self.toggle_optimizer(optG)
            optG.zero_grad()

            mu, log_var, z = self.model.vae_forward(real_data)
            fake_data = self.model(z)
           
            # Loss measures generator's ability to fool the discriminator,Train on fake images
            recon_loss = F.mse_loss(fake_data, real_data, reduction='sum') / batch_size
            fake_validity  = critic(fake_data)
    
            errG = -torch.mean(fake_validity) + self.recon_weight * recon_loss
        
            self.manual_backward(errG)
            optG.step()

            self.log("generator/fake_validity", -fake_validity.mean())
            self.log("generator/recon_loss", recon_loss)
            self.log("generator/total_loss", errG, prog_bar=True)
  
            self.untoggle_optimizer(optG) 

    def validation_step(self, batch, batch_idx):
        
        img, _ = batch
        
        N = img.size(0)

        _, _, z = self.model.vae_forward(img) 
        fake_imgs = self.model(z)

        val_mse_sum = F.mse_loss(img, fake_imgs, reduction='sum') / N

        self.log("metrics/val_mse_reduction", val_mse_sum, on_epoch=True)
    
        return ValidationResult(real_image=img, fake_image=fake_imgs, encode_latent=z)

  
    def configure_optimizers(self):
        lrE = 0.0003  # Learning rate for the encoder
        lrG = 0.01  # Learning rate for the generator
        lrD = 0.0005  # Learning rate for the discriminator
    
        match self.optimizer.lower():
            case "adam":

                b1 = 0
                b2 = 0.9

                optE = torch.optim.Adam(
                    self.model.encoder.parameters(),
                    lr=lrE,
                    weight_decay=self.weight_decay,
                    betas=(0, b2),
                )

                optG = torch.optim.Adam(
                    self.model.generator.parameters(),
                    lr=lrG,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

                optD = torch.optim.Adam(
                    self.model.discriminator.parameters(),
                    lr=lrD,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2)
                )
            case "sgd":

                optE = torch.optim.SGD(
                    self.model.encoder.parameters(),
                    lr = lrG)
                
                optG = torch.optim.SGD(
                    self.model.generator.parameters(),
                    lr=lrG)
                
                optD = torch.optim.SGD(
                    self.model.discriminator.parameters(), 
                    lr=lrD)
                
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")
            
        return [optE,optG, optD], []

        

       