import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
from abc import ABC, abstractmethod
# from deepspeed.ops.adam import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torchmetrics.functional import accuracy
from torchmetrics import Accuracy, MeanMetric


class WrapperBase(pl.LightningModule, ABC):
    def __init__(
        self,
        model,
        learning_rate=5e-4,
        weight_decay=0.0,
        epochs=1,
        optimizer=None,
        dataset_info=None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)
    
    @abstractmethod
    def training_step(self, batch):
        pass
    
    @abstractmethod
    def configure_optimizers(self):
        pass