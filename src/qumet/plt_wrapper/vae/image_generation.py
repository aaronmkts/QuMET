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
    
    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(sample)

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
    
    def loss_function(self, mu, std, x, x_hat, z):
        #Reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.model.log_scale, x)   
        #KL divergence
        kl = self.kl_divergence(z, mu, std)
        #ELBO
        elbo = kl - recon_loss

        return {'elbo': elbo.mean(),
                'kl': kl.mean(),
                'recon_loss': recon_loss.mean()}
    
    def training_step(self, batch):

        x, _ = batch

        mu, std, x_hat, z = self.model.forward(x)

        total_loss = self.loss_function(mu, std, x, x_hat, z)

        self.log('train_kl_loss', total_loss['kl'], on_step=True,
                    on_epoch=True, prog_bar=False)
        self.log('train_recon_loss', total_loss['recon_loss'], on_step=True,
                    on_epoch=True, prog_bar=False)
        self.log('train_loss', total_loss['elbo'], on_step=True,
                    on_epoch=True, prog_bar=True)
 
        return total_loss['elbo']

    def validation_step(self, batch):

        x, _ = batch
        
        mu, std, x_hat, z = self.model.forward(x)
        total_loss = self.loss_function(mu, std, x, x_hat, z)
        self.log('val_loss', total_loss['elbo'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_kl_epoch', total_loss['kl'], on_step=True, on_epoch=True, prog_bar=True)

        self.logger.experiment.add_image('Normalized Inputs', make_grid(x[:8]))
