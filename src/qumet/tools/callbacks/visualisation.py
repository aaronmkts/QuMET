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