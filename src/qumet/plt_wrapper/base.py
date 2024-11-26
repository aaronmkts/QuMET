import lightning.pytorch as pl
import torch
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
# from torchmetrics.functional import accuracy
from torchmetrics import Accuracy, MeanMetric
from dataclasses import dataclass, field

class WrapperBase(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=5e-4,
        weight_decay=0.0,
        epochs=1,
        optimizer=None,
        dataset_info=None,
        freeze_modules=None
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer
        self.dataset_info = dataset_info
        self.freeze_modules=freeze_modules

        if self.freeze_modules is not None:
            freeze_networks = set()
                
            for item in self.freeze_modules:
                network = item.split('.')[0]
                freeze_networks.add(network)
            
            freeze_networks = list(freeze_networks)
            self.freeze_networks = freeze_networks
        else:
            self.freeze_networks = None  
        

    def forward(self, x):
        raise NotImplementedError()
    
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

@dataclass
class ValidationResult():
    others: dict = field(default_factory=dict)
    real_image: torch.Tensor = None
    fake_image: torch.Tensor = None
    recon_image: torch.Tensor = None
    label: torch.Tensor = None
    encode_latent: torch.Tensor = None
    real_distribution: torch.Tensor = None
    fake_distribution: torch.Tensor = None
