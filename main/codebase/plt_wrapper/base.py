import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
# from deepspeed.ops.adam import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
# from torchmetrics.functional import accuracy
from torchmetrics import Accuracy, MeanMetric


class WrapperBase(pl.LightningModule):
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
    
    def training_step(self, batch):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def plot_to_image(self,figure ):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        return im
