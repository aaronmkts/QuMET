
from lightning import LightningModule, Trainer
import numpy as np
from lightning.pytorch.callbacks import Callback
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from qumet.plt_wrapper.base import ValidationResult

class GANImagesCallback(Callback):
    def __init__(self, batch_size=64, every_n_epochs=1, nrow=8):
        """
        Args:
            batch_size (int): Number of images to sample and display.
            every_n_epochs (int): Frequency of image logging (in epochs).
            nrow (int): Number of images per row in the grid.
            save_images (bool): Whether to save the generated images to disk.
        """
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.nrow = nrow

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx):
        if trainer.current_epoch % self.every_n_epochs == 0 and batch_idx == 0:
            # Real images
            real_grid = get_grid_images(outputs.real_image, self.batch_size, self.nrow)
            trainer.logger.experiment.add_image("images/real", real_grid, global_step=trainer.current_epoch)

            # Reconstructed images (if available)
            if outputs.recon_image is not None:
                recon_grid = get_grid_images(outputs.recon_image, self.batch_size, self.nrow)
                trainer.logger.experiment.add_image("images/recon", recon_grid, global_step=trainer.current_epoch)

            # Fake images
            if outputs.fake_image is not None:
                fake_grid = get_grid_images(outputs.fake_image, self.batch_size, self.nrow)
                trainer.logger.experiment.add_image("images/sample", fake_grid, global_step=trainer.current_epoch)


            # Additional custom outputs
            if hasattr(outputs, "others") and outputs.others:
                for key, img in outputs.others.items():
                    if img is not None:
                        grid = get_grid_images(img, self.batch_size, self.nrow)
                        trainer.logger.experiment.add_image(f"images/{key}", grid, global_step=trainer.current_epoch)

def get_grid_images(imgs, nimgs=64, nrow=8):
    """Create a grid of images for visualization."""
    grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=True, nrow=nrow, pad_value=1)
    return grid


''' 

class BarsStripesCallback(Callback):
    def __init__(self, every_n_epochs=99):
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,  outputs: ValidationResult, batch, batch_idx) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            plotBars(outputs.real_distribution, outputs.fake_distribution)



def plotBars(real_dist, fake_dist):
    size = np.log2(len(real_dist))
    plt.figure(figsize=(12, 5))

    plt.bar(
        np.arange(2**size), real_dist, width=2.0, label=r"$\pi(x)$", alpha=0.4, color="tab:blue"
    )

    plt.bar(
        np.arange(2**size),
        fake_dist,
        width=2.0,
        label=r"$p_\theta(x)$",
        alpha=0.9,
        color="tab:green",
    )

    # Hide x-ticks
    plt.xticks([], fontsize=16)
    plt.xlabel("Samples", fontsize=16)
    plt.ylabel("Prob. Distribution", fontsize=16)

    # Set y-tick fontsize
    plt.yticks(fontsize=16)

    plt.legend(loc="upper right", fontsize=16)
    plt.subplots_adjust(bottom=0.3)
    plt.show()






class VAEImageSampler(Callback):
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

'''