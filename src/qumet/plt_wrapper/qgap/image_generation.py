import torch
import torch.nn as nn
import numpy as np
import math
from ..base import WrapperBase, ValidationResult
from abc import abstractmethod
import torch.nn.functional as F
import torch.nn.functional as F


class GAP_UNET(WrapperBase):
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

        self.dataset_info = dataset_info
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


    def photonLoss(self,result, target):
        expEnergy = torch.exp(result)
        perImage =  -torch.mean(result*target, dim =(-1,-2,-3), keepdims = True )
        perImage += torch.log(torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))*torch.mean(target, dim =(-1,-2,-3), keepdims = True )
        return torch.mean(perImage)
    
    def MSELoss(self,result, target):
        expEnergy = torch.exp(result)
        expEnergy /= (torch.mean(expEnergy, dim =(-1,-2,-3), keepdims = True ))
        target = target / (torch.mean(target, dim =(-1,-2,-3), keepdims = True ))
        return torch.mean((expEnergy-target)**2)
    
    def training_step(self, batch, batch_idx):
        loss = self.photonLoss(self(batch[:,self.model.channels:,...]),batch[:,:self.model.channels,...] )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.photonLoss(self(batch[:,self.model.channels:,...]),batch[:,:self.model.channels,...] )
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.photonLoss(self(batch[:,self.model.channels:,...]),batch[:,:self.model.channels,...] )
        self.log("test_loss", loss)
        
    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training 
        match self.optimizer.lower():
            case "adam":
                b1 = 0.9
                b2 = 0.999

                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(b1, b2),
                )

            case "sgd":

                optimizer = torch.optim.SGD(
                    self.model.parameters(), 
                    lr=self.learning_rate,)

            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        return { 'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_loss'
        }