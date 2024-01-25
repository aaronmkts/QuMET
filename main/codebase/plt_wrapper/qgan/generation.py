import torch
import torch.nn as nn
from torchmetrics import Accuracy, MeanMetric
from torch.optim.lr_scheduler import CosineAnnealingLR

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

    def training_step(self, batch, batch_idx):
        
        
        
        
    
        return 
    
    def configure_optimizers(self):
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        match self.optimizer: 
            case "adamw":

                optG = torch.optim.AdamW(
                    self.trainer.model.generator.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

                schedulerG = CosineAnnealingLR(optG, T_max=self.epochs, eta_min=1e-6)
                
                optD = torch.optim.AdamW(
                    self.trainer.model.discriminator.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

                schedulerD = CosineAnnealingLR(optD, T_max=self.epochs, eta_min=1e-6)


            case  "adam":
                opt = torch.optim.Adam(
                    self.trainer.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
                scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)

            case "sgd_warmup" | "sgd":
                opt = torch.optim.SGD(
                    self.trainer.model.parameters(),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=0.0005,
                    nesterov=True,
                )
                if self.optimizer == "sgd":
                    scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=0.0)
        
            case _:
                raise ValueError(f"Unsupported optimizer name {self.optimizer}")
        return {"optimizerG": optG, "lr_schedulerG": schedulerG}