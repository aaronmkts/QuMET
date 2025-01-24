import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Dict, Any
from ..base import ValidationResult
from ..utils import compute_gradient_penalty

class VAEGANWrapper(pl.LightningModule):
    """
    A generic VAE-GAN wrapper that decouples the VAE functionality
    (encoder, reparameterization, VAE losses) from any given GAN model.
    
    Requirements for `base_model`:
      - Must have `base_model.generator` (nn.Module)
      - Must have `base_model.discriminator` (nn.Module)
      
    The `encoder` can be a separate module or part of `base_model`.
    """
    def __init__(
        self,
        base_model: nn.Module,
        encoder: nn.Module,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0.0,
        epochs=100,
        optimizer=None):
        """
        Args:
            base_model: A GAN model (with .generator and .discriminator).
            encoder: An encoder module that returns (mu, log_var) from an input image.
            config: A dictionary containing hyperparameters or other user configs.
            learning_rates: Dict with possible keys {lrE, lrG, lrD}. Defaults if not provided.
            weight_decay: Weight decay used by optimizers.
            recon_weight: Weight for reconstruction loss in generator training.
            lambda_gp: Gradient penalty weight (if using WGAN-GP).
            n_critic: Number of critic (discriminator) steps per generator step.
            optimizer_type: Type of optimizer to use ("adam", "sgd", etc.).
        """
   
        super().__init__()
        self.save_hyperparameters(ignore=["base_model", "encoder"])  # (optional) track hyperparams
        self.automatic_optimization = False
        self.base_model = base_model
        self.encoder = encoder
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.recon_weight = 5e-4
        self.lambda_gp = 10
        self.n_critic = 5

        self.lrE = 0.0003
        self.lrG = 0.01
        self.lrD = 0.0005

        # Extract any needed shape info from config
        self.z_dim = self.encoder.z_dim
        self.image_shape = dataset_info.image_size
        
    def forward(self, z):

        return self.base_model.generator(z)

    def encode(self, x):
        """
        Encode an image into (mu, log_var) using the provided encoder.
        """
        mu, log_var = self.encoder(x)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        """
        Reparameterization trick to sample a latent vector z.
        """
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def normal_kl_div(self, mu, log_var):
        """
        Compute KL divergence for standard normal prior.
        """
        prior_loss = 1 + log_var - mu.pow(2) - log_var.exp()
        kl_divergence = torch.mean(-0.5 * torch.sum(prior_loss, dim = 1))
        return kl_divergence

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step for the VAE-GAN, updating the discriminator, encoder, and generator.

        This method processes a batch of real data and executes the following steps:
        
        1. **Discriminator Update:**
        - Encodes real data to obtain latent vectors (mu, log_var).
        - Samples latent vectors `z` using the reparameterization trick.
        - Generates fake data using the generator from sampled `z`.
        - Passes both real and fake data through the discriminator.
        - Computes adversarial loss with gradient penalty.
        - Backpropagates and updates discriminator weights.
        - Logs discriminator loss and Wasserstein distance.
        
        2. **Encoder and Generator Update (Conditionally):**
        - Every `n_critic` steps, the encoder and generator are updated:
            
            a. **Encoder Update:**
                - Re-encodes real data and regenerates fake data.
                - Computes reconstruction loss (MSE) between reconstructed and real data.
                - Computes KL divergence between the encoded distribution and the standard normal prior.
                - Combines reconstruction and KL divergence losses.
                - Backpropagates and updates encoder weights.
                - Logs encoder-specific losses.
            
            b. **Generator Update:**
                - Re-encodes real data and regenerates fake data.
                - Passes fake data through the discriminator.
                - Computes generator adversarial loss to fool the discriminator.
                - Adds a weighted reconstruction loss to the generator loss.
                - Backpropagates and updates generator weights.
                - Logs generator-specific losses.

        Returns:
            torch.Tensor: The loss value for the current training step, used for logging purposes.
        """

        # data and real/fake labels
        real_data, _ = batch
        batch_size = real_data.size(0)
        optE, optG, optD = self.optimizers()
  
        # Generate fake-data using noise input
        mu, log_var = self.encode(real_data)
        z = self.reparametrize(mu, log_var)
        fake_data = self.base_model.generator(z)
        
        # Training the discriminator
        self.toggle_optimizer(optD)
        optD.zero_grad()
            
        real_validity  = self.base_model.discriminator(real_data)
        fake_validity  = self.base_model.discriminator(fake_data.detach())

            # Adversarial loss
        gradient_penalty = compute_gradient_penalty(self.base_model.discriminator, real_data, fake_data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

        wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)

        self.log("discriminator/total_loss", d_loss)
        self.log("wasserstein_distance", wasserstein_distance, prog_bar=True)

        self.manual_backward(d_loss)
        optD.step()
        self.untoggle_optimizer(optD)

        if batch_idx!= 0 and batch_idx % (self.n_critic) == 0:
        
            # Train the encoder
            self.toggle_optimizer(optE)
            optE.zero_grad()
            
            # Recompute recon_imgs for the encoder
            mu, log_var = self.encode(real_data)
            z = self.reparametrize(mu, log_var)
            recon_imgs = self.base_model.generator(z)
            
            recon_loss = F.mse_loss(recon_imgs, real_data, reduction='sum') / batch_size
            prior_loss = self.normal_kl_div(mu, log_var) 

            errE =  prior_loss + recon_loss
            self.manual_backward(errE)
            optE.step()

            self.log('encoder/prior_loss', prior_loss)
            self.log('encoder/recon_loss', recon_loss)
            self.log('encoder/total_loss', errE, prog_bar=True)

            self.untoggle_optimizer(optE)

            self.toggle_optimizer(optG)
            optG.zero_grad()

            mu, log_var = self.encode(real_data)
            z = self.reparametrize(mu, log_var)
            fake_data = self.base_model.generator(z)
           
            # Loss measures generator's ability to fool the discriminator,Train on fake images
            recon_loss = F.mse_loss(fake_data, real_data, reduction='sum') / batch_size
            fake_validity = self.base_model.discriminator(fake_data)
    
            g_loss = -torch.mean(fake_validity) + self.recon_weight * recon_loss
        
            self.manual_backward(g_loss)
            optG.step()

            self.log("generator/fake_validity", -fake_validity.mean())
            self.log("generator/recon_loss", recon_loss)
            self.log("generator/total_loss", g_loss, prog_bar=True)
  
            self.untoggle_optimizer(optG) 
    
    def validation_step(self, batch, batch_idx):
        '''
        Validation step for the model
        '''
        
        real_imgs, labels = batch
        
        mu, log_var = self.encode(real_imgs)
        z = self.reparametrize(mu, log_var)
        fake_imgs = self.base_model.generator(z)
        
        N = real_imgs.size(0)
        val_recon_loss = F.mse_loss(real_imgs, fake_imgs, reduction='sum') / N

        self.log('metrics/val_mse_reduction', val_recon_loss, on_epoch=True)
    
        return ValidationResult(real_image=real_imgs, fake_image=fake_imgs, encode_latent=z, label=labels)

    def configure_optimizers(self):
        """
        Set up three optimizers: for encoder, generator, discriminator.
        """
        if self.optimizer == "adam":
            optE = torch.optim.Adam(
                self.encoder.parameters(),
                lr=self.lrE,
                weight_decay=self.weight_decay,
                betas=(0.0, 0.9)
            )
            optG = torch.optim.Adam(
                self.base_model.generator.parameters(),
                lr=self.lrG,
                weight_decay=self.weight_decay,
                betas=(0.0, 0.9)
            )
            optD = torch.optim.Adam(
                self.base_model.discriminator.parameters(),
                lr=self.lrD,
                weight_decay=self.weight_decay,
                betas=(0.0, 0.9)
            )
        elif self.optimizer == "sgd":
            optE = torch.optim.SGD(self.encoder.parameters(), lr=self.lrE, weight_decay=self.weight_decay)
            optG = torch.optim.SGD(self.base_model.generator.parameters(), lr=self.lrG, weight_decay=self.weight_decay)
            optD = torch.optim.SGD(self.base_model.discriminator.parameters(), lr=self.lrD, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type {self.optimizer_type}")

        return [optE, optG, optD], []

