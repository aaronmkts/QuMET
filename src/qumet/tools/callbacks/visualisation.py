
from lightning import LightningModule, Trainer
import numpy as np
from lightning.pytorch.callbacks import Callback
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from qumet.plt_wrapper.base import ValidationResult

class SampleImagesCallback(Callback):
    def __init__(self, batch_size=64, every_n_epochs=1):
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx):
        if trainer.current_epoch % self.every_n_epochs == 0 and batch_idx == 0:
            result_path = Path("results")
            result_path.mkdir(parents=True, exist_ok=True)

            real_grid = get_grid_images(outputs.real_image, pl_module)
            trainer.logger.experiment.add_image("images/real", real_grid, global_step=trainer.current_epoch)

            if outputs.recon_image is not None:
                recon_grid = get_grid_images(outputs.recon_image, pl_module)
                trainer.logger.experiment.add_image("images/recon", recon_grid, global_step=trainer.current_epoch)

            if outputs.fake_image is not None:
                fake_grid = get_grid_images(outputs.fake_image, pl_module)
                trainer.logger.experiment.add_image("images/sample", fake_grid, global_step=trainer.current_epoch)
                #torchvision.utils.save_image(fake_grid, result_path / f"{trainer.current_epoch}.jpg")

            for key in outputs.others:
                if outputs.others[key] is not None:
                    grid = get_grid_images(outputs.others[key], pl_module)
                    trainer.logger.experiment.add_image(f"images/{key}", grid, global_step=trainer.current_epoch)

def get_grid_images(imgs, model, nimgs=64, nrow=8):
    grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=True, nrow=nrow, pad_value=1)
    return grid


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

