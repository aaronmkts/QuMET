from lightning.pytorch.callbacks import Callback
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from qumet.plt_wrapper.base import ValidationResult
import lightning.pytorch as pl
'''
class FIDEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1):
        self.every_n_epoch = every_n_epochs

    def image_float2int(self, imgs, pl_module):
        if pl_module.input_normalize:
            imgs = (imgs + 1) / 2
        imgs = (imgs * 255).to(torch.uint8)
        return imgs

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            self.fid = FrechetInceptionDistance().to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx):
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            real_imgs, fake_images = outputs.real_image, outputs.fake_image
            self.fid.update(self.image_float2int(real_imgs, pl_module), real=True)
            self.fid.update(self.image_float2int(fake_images, pl_module), real=False)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if pl_module.channels == 3 and trainer.current_epoch % self.every_n_epoch == 0:
            pl_module.log("metrics/fid", self.fid.compute(), on_epoch=True)
'''
class FIDEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1):
        self.every_n_epoch = every_n_epochs

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch % self.every_n_epoch == 0:
            self.fid = FrechetInceptionDistance(normalize=True).to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx = 0):
        if trainer.current_epoch % self.every_n_epoch == 0:

            real_imgs, fake_images = outputs.real_image, outputs.fake_image
            # Convert grayscale images to three-channel images
            real_imgs_3channel, fake_images_3channel = real_imgs.repeat(1, 3, 1, 1), fake_images.repeat(1, 3, 1, 1)
            self.fid.update(real_imgs_3channel, real=True)
            self.fid.update(fake_images_3channel, real=False)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.every_n_epoch == 0:
            pl_module.log("metrics/fid", self.fid.compute(), on_epoch=True)



class ISEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1):
        self.every_n_epoch = every_n_epochs

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch % self.every_n_epoch == 0:
            self.inception_score = InceptionScore(normalize=True).to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx = 0):
        if trainer.current_epoch % self.every_n_epoch == 0:
            fake_images = outputs.fake_image
            # Convert grayscale images to three-channel images
            fake_images_3channel = fake_images.repeat(1, 3, 1, 1)
            self.inception_score.update(fake_images_3channel)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.every_n_epoch == 0:
            inception_mean = self.inception_score.compute()[0]
            pl_module.log("metrics/is", inception_mean, on_epoch=True)  