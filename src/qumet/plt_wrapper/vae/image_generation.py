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
from torchvision.utils import save_image, make_grid

class VAEImageGenerationModelWrapper(WrapperBase):
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
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_preds = 16

    @abstractmethod
    def training_step(self, batch):
        pass


    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        match self.optimizer.lower():
            case "adam":
                b1 = 0.777
                b2 = 0.999

                optG = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

            case "sgd":

                optG = torch.optim.SGD(self.model.parameters(),
                                        lr=self.learning_rate)
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")

        return [optG], []
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        return log_pxz.sum(dim=(1, 2, 3))

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
  
class VAEWrapper(VAEImageGenerationModelWrapper):
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

        x, _ = batch

        mu, std, z, x_hat = self.model.forward(x)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.model.log_scale, x)

        #expectation under z of the kl divergence between q(z|x) and
        #a standard normal distribution of the same shape
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
    
        elbo = elbo.mean()

        self.log('train_kl_loss', kl.mean(), on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss.mean(), on_step=True,
                 on_epoch=True, prog_bar=False)
        self.log('train_loss', elbo, on_step=True,
                 on_epoch=True, prog_bar=True)

        # train_images = make_grid(x[:16]).cpu().numpy()
        return elbo

    def validation_step(self, batch):

        x, _ = batch
        
        mu, std, z, x_hat = self.model.forward(x)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.model.log_scale, x)

        #expectation under z of the kl divergence between q(z|x) and
        #a standard normal distribution of the same shape
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        self.log('val_kl_epoch', kl.mean(), on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss.mean(), on_step=False, on_epoch=True)
        self.log('val_loss', elbo, on_step=False, on_epoch=True)

        self.logger.experiment.add_image('Normalized Inputs', make_grid(x[:8]))


        return x_hat, elbo