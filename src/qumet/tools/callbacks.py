from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import Callback
import torch
from collections import namedtuple


class ImageSampler(Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        INTERVAL = 50
        if batch_idx % INTERVAL == 0:
            # Z COMES FROM NORMAL(0, 1)
            sample_shape = (self.num_preds, pl_module.model.z_dim)
            p = torch.distributions.Normal(torch.zeros(sample_shape), torch.ones(sample_shape))
            z = p.rsample()

            # SAMPLE IMAGES
            with torch.no_grad():
                pred = pl_module.model.decoder(z.to(pl_module.device)).cpu()
            # CONVERT IMAGES TO GRID
            img = make_grid(pred).permute(1, 2, 0).numpy()

            # PLOT IMAGES
            trainer.logger.experiment.add_image(
                'Sample Images',
                torch.tensor(img).permute(2, 0, 1),
                global_step=trainer.global_step
            )
