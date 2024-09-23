from lightning.pytorch.callbacks import Callback
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from qumet.plt_wrapper.base import ValidationResult
import lightning.pytorch as pl
import torch
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from ...plt_wrapper.metrics import NDB
import torchvision
import joblib
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

class GMMEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1):
        self.every_n_epoch = every_n_epochs
        self.z_samples = []
        self.real_images = []

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx=0):
        if trainer.current_epoch == trainer.max_epochs - 1:

            z = outputs.encode_latent
            real_images = outputs.real_image

            self.z_samples.append(z)
            self.real_images.append(real_images)
            

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch == trainer.max_epochs - 1:
            real_images = torch.cat(self.real_images, dim=0)
            z_samples = torch.cat(self.z_samples, dim=0).numpy()

            #DATA Instance Prior, Gaussion Mixture Model 
            N = z_samples.shape[0]
            gmm = GaussianMixture(n_components=50, random_state=9).fit(z_samples)
            joblib.dump(gmm, 'gmm_model_seed9_mnist.pkl')
            disp_prior, _ = gmm.sample(N)
            disp_prior = torch.tensor(disp_prior, dtype= torch.float32).to(pl_module.device)
            fake_images = pl_module.model(disp_prior)

            # MSE Evaluation

            val_mse_disp = F.mse_loss(real_images, fake_images, reduction='sum') / N
            pl_module.log("val_log/val_mse_disp", val_mse_disp, on_epoch=True)

            # JSD, NDB Evaluation
            real_images_flat = real_images.reshape(real_images.size(0), -1).numpy()
            fake_images_flat = fake_images.reshape(fake_images.size(0), -1).numpy()

            ndb_k30 = NDB(training_data=real_images_flat, number_of_bins= 30, whitening=False, z_threshold=4)
            ndb_k50 = NDB(training_data=real_images_flat, number_of_bins= 50, whitening=False, z_threshold=4)

            results_k50 = ndb_k50.evaluate(fake_images_flat)
            results_k30 = ndb_k30.evaluate(fake_images_flat)

            ndb_k50 = float(results_k50["NDB"])/ndb_k50.number_of_bins
            jsd_50 = float(results_k50["JS"])

            pl_module.log('val_log/val_ndb_k50_disp', ndb_k50, on_epoch=True)
            pl_module.log('val_log/val_jsd_k50_disp', jsd_50, on_epoch=True)

            ndb_k30 = float(results_k30["NDB"])/ndb_k30.number_of_bins
            jsd_k30 = float(results_k30["JS"])

            pl_module.log('val_log/val_ndb_k30_disp', ndb_k30, on_epoch=True)   
            pl_module.log('val_log/val_jsd_k30_disp', jsd_k30, on_epoch=True)
            print('GMM eval complete', 'NDB 30:', ndb_k30, 'JSD 30:', jsd_k30, 'NDB 50:', ndb_k50, 'JSD 50:', jsd_50, 'MSE:', val_mse_disp)
            # Log fake images for visualization 

            fake_grid = get_grid_images(fake_images, pl_module)
            trainer.logger.experiment.add_image("images/disp_prior", fake_grid, global_step=trainer.current_epoch)
          
            self.real_images.clear()
            self.z_samples.clear()
           

def get_grid_images(imgs, model, nimgs=8, nrow=8):
    grid = torchvision.utils.make_grid(imgs[:nimgs], normalize=True, nrow=nrow, pad_value=1)
    return grid