from lightning.pytorch.callbacks import Callback
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from qumet.plt_wrapper.base import ValidationResult
import lightning.pytorch as pl
import torch
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import wandb
from ...plt_wrapper.metrics import NDB_JSD_Metric
from .visualisation import GANImagesCallback
import torchvision
import joblib
import numpy as np 
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio 
# General GAN Callbacks
'''
class FIDEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1, feature = 2048, reset_real_features=False, 
                 normalize=True,input_img_size=(3, 299, 299)):
        """
        Args:
            every_n_epochs (int): How often to compute the FID (in epochs).
            normalize (bool): Whether to normalize the input images.
        """
        self.every_n_epoch = every_n_epochs
        self.feature = feature
        self.reset_real_features = reset_real_features
        self.normalize = normalize
        self.input_img_size = input_img_size
        self.fid = None

    def setup_fid(self, pl_module):
        """Initialize the FID metric on the current device."""
        self.fid = FrechetInceptionDistance(feature=self.feature, normalize=self.normalize,
                                            reset_real_features=self.reset_real_features, 
                                            input_img_size = self.input_img_size).to(pl_module.device)

    def convert_to_3channel(self, images):
        """Convert grayscale images to 3-channel images."""
        return images.repeat(1, 3, 1, 1)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Set up the FID metric at the start of the validation epoch."""
        if trainer.current_epoch % self.every_n_epoch == 0:
            self.setup_fid(pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, 
                                batch, batch_idx, dataloader_idx=0):
        """Update FID with real and fake images during validation."""
        if trainer.current_epoch % self.every_n_epoch == 0 and outputs:
            real_imgs, fake_imgs = outputs.real_image, outputs.fake_image
            
            if real_imgs is not None and fake_imgs is not None:
                real_imgs_3channel = self.convert_to_3channel(real_imgs)
                fake_imgs_3channel = self.convert_to_3channel(fake_imgs)
                self.fid.update(real_imgs_3channel, real=True)
                self.fid.update(fake_imgs_3channel, real=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log the FID score at the end of the validation epoch."""
        if trainer.current_epoch % self.every_n_epoch == 0:
            if self.fid is not None:
                fid_score = self.fid.compute()
                pl_module.log("metrics/fid", fid_score, on_epoch=True)
                self.fid.reset()  # Reset the metric for the next epoch
''' 

class PSNRCallback(Callback):
    def __init__(self, every_n_epochs=1):
        """
        Args:
            every_n_epochs (int): How often to compute the FID (in epochs).
        """
        super().__init__()
        self.every_n_epoch = every_n_epochs
        
        # Buffers to store real/fake images each epoch
        self.real_images_accum = []
        self.fake_images_accum = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect the real and fake images at the end of each validation batch.
        """
        if trainer.current_epoch % self.every_n_epoch == 0:
            if outputs is not None:
                real_imgs = getattr(outputs, 'real_image', None)
                fake_imgs = getattr(outputs, 'fake_image', None)
                
                if real_imgs is not None and fake_imgs is not None:
                    self.real_images_accum.append(real_imgs)
                    self.fake_images_accum.append(fake_imgs)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Compute the manual PSNR score over all accumulated images.
        """
  
        if trainer.current_epoch % self.every_n_epoch == 0:
      
            if len(self.real_images_accum) > 0 and len(self.fake_images_accum) > 0:
                # Concatenate all real and fake images along the batch dimension
                real_images = torch.cat(self.real_images_accum, dim=0)
                fake_images = torch.cat(self.fake_images_accum, dim=0)
                
                real = fake_images.detach().cpu().numpy().reshape(-1, 28, 28)
                fake = real_images.detach().cpu().numpy().reshape(-1, 28, 28)

                psnr_list = []
                
                for i in range(len(real)):
                    psnr_val = peak_signal_noise_ratio(real[i], fake[i])
                    psnr_list.append(psnr_val)

                psnr_mean = np.mean(psnr_list)
                
                # Log the result
                pl_module.log("metrics/psnr", psnr_mean, on_epoch=True)

            # Clear buffers or do any post-processing here
            self.real_images_accum.clear()
            self.fake_images_accum.clear()

class SSIMCallback(Callback):
    def __init__(self, every_n_epochs=1):
        """
        Args:
            every_n_epochs (int): How often to compute the FID (in epochs).
        """
        super().__init__()
        self.every_n_epoch = every_n_epochs
        
        # Buffers to store real/fake images each epoch
        self.real_images_accum = []
        self.fake_images_accum = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect the real and fake images at the end of each validation batch.
        """
        if trainer.current_epoch % self.every_n_epoch == 0:
            if outputs is not None:
                real_imgs = getattr(outputs, 'real_image', None)
                fake_imgs = getattr(outputs, 'fake_image', None)
                
                if real_imgs is not None and fake_imgs is not None:
                    self.real_images_accum.append(real_imgs)
                    self.fake_images_accum.append(fake_imgs)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Compute the manual FID score over all accumulated images.
        """
  
        if trainer.current_epoch % self.every_n_epoch == 0:
      
            if len(self.real_images_accum) > 0 and len(self.fake_images_accum) > 0:
                # Concatenate all real and fake images along the batch dimension
                real_images = torch.cat(self.real_images_accum, dim=0)
                fake_images = torch.cat(self.fake_images_accum, dim=0)
                
                fake = fake_images.detach().cpu().numpy().reshape(-1, 28, 28)
                real= real_images.detach().cpu().numpy().reshape(-1, 28, 28)

                ssim_values = []
                for i in range(len(real)):
                    ssim_val = structural_similarity(
                        real[i], 
                        fake[i],
                        data_range=1.0  
                    )
                    ssim_values.append(ssim_val)

                ssim_mean = np.mean(ssim_values)
                
                # Log the result
                pl_module.log("metrics/ssim", ssim_mean, on_epoch=True)

            self.real_images_accum.clear()
            self.fake_images_accum.clear()
class CosSimilarityEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1):
        """
        Args:
            every_n_epochs (int): How often to compute the FID (in epochs).
        """
        super().__init__()
        self.every_n_epoch = every_n_epochs
        
        # Buffers to store real/fake images each epoch
        self.real_images_accum = []
        self.fake_images_accum = []

    def calculate_cos(self, v1, v2):
        v1 = v1.detach().cpu().numpy().reshape(-1, 784)
        v2 = v2.detach().cpu().numpy().reshape(-1, 784)
        num = np.dot(v1, np.array(v2).T) 
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1) 
        res = num / denom
        res[np.isneginf(res)] = 0
        res = 0.5 + 0.5 * res
        cos_mean = np.mean(res)
        return cos_mean
   

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect the real and fake images at the end of each validation batch.
        """
        if trainer.current_epoch % self.every_n_epoch == 0:
            if outputs is not None:
                real_imgs = getattr(outputs, 'real_image', None)
                fake_imgs = getattr(outputs, 'fake_image', None)
                
                if real_imgs is not None and fake_imgs is not None:
                    self.real_images_accum.append(real_imgs)
                    self.fake_images_accum.append(fake_imgs)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Compute the manual cosine similarity score over all accumulated images.
        """
  
        if trainer.current_epoch % self.every_n_epoch == 0:
      
            if len(self.real_images_accum) > 0 and len(self.fake_images_accum) > 0:

                real_images = torch.cat(self.real_images_accum, dim=0)
                fake_images = torch.cat(self.fake_images_accum, dim=0)
                
                cos_sim_score = self.calculate_cos(real_images, fake_images)
                pl_module.log("metrics/cos_sim", cos_sim_score, on_epoch=True)

            # Clear buffers or do any post-processing here
            self.real_images_accum.clear()
            self.fake_images_accum.clear()

class FIDEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1):
        """
        Args:
            every_n_epochs (int): How often to compute the FID (in epochs).
        """
        super().__init__()
        self.every_n_epoch = every_n_epochs
        
        # Buffers to store real/fake images each epoch
        self.real_images_accum = []
        self.fake_images_accum = []

    def calculate_fid(self, act1, act2):
        """
        Compute FID given two sets of activations or flattened images.
        By default, this is set up for 28x28 images => 784-dim. 
        Adjust as needed if your images have different shape.
        """
        # Move to CPU and flatten
        act1 = act1.detach().cpu().numpy().reshape([-1, 784])
        act2 = act2.detach().cpu().numpy().reshape([-1, 784])

        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)

        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Frechet Distance
        fid_value = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid_value
   

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect the real and fake images at the end of each validation batch.
        """
        if trainer.current_epoch % self.every_n_epoch == 0:
            if outputs is not None:
                real_imgs = getattr(outputs, 'real_image', None)
                fake_imgs = getattr(outputs, 'fake_image', None)
                
                if real_imgs is not None and fake_imgs is not None:
                    self.real_images_accum.append(real_imgs)
                    self.fake_images_accum.append(fake_imgs)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Compute the manual FD score over all accumulated images.
        """
  
        if trainer.current_epoch % self.every_n_epoch == 0:
      
            if len(self.real_images_accum) > 0 and len(self.fake_images_accum) > 0:
                # Concatenate all real and fake images along the batch dimension
                real_images = torch.cat(self.real_images_accum, dim=0)
                fake_images = torch.cat(self.fake_images_accum, dim=0)
                
                fid_score = self.calculate_fid(real_images, fake_images)
                
                # Log the result
                pl_module.log("metrics/fid", fid_score, on_epoch=True)

            # Clear buffers or do any post-processing here
            self.real_images_accum.clear()
            self.fake_images_accum.clear()

class ISEvaluationCallback(Callback):
    def __init__(self, every_n_epochs=1, feature = 'logits_unbiased', splits = 10, normalize=True):
        """
        Args:
            every_n_epochs (int): How often to compute the IS (in epochs).
            normalize (bool): Whether to normalize the input images.
        """
        self.every_n_epoch = every_n_epochs
        self.feature = feature
        self.splits = splits
        self.normalize = normalize
        self.inception_score = None

    def setup_inception_score(self, pl_module):
        """Initialize the Inception Score metric on the current device."""
        self.inception_score = InceptionScore(feature=self.feature, splits =self.splits, normalize=self.normalize).to(pl_module.device)

    def convert_to_3channel(self, images):
        """Convert grayscale images to 3-channel images."""
        return images.repeat(1, 3, 1, 1)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Set up the Inception Score metric at the start of the validation epoch."""
        if trainer.current_epoch % self.every_n_epoch == 0:
            self.setup_inception_score(pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx, dataloader_idx=0):
        """Update Inception Score with fake images during validation."""
        if trainer.current_epoch % self.every_n_epoch == 0 and outputs:
            fake_imgs = outputs.fake_image
            
            if fake_imgs is not None:
                fake_imgs_3channel = self.convert_to_3channel(fake_imgs)
                self.inception_score.update(fake_imgs_3channel)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log the Inception Score at the end of the validation epoch."""
        if trainer.current_epoch % self.every_n_epoch == 0:
            if self.inception_score is not None:
                inception_mean = self.inception_score.compute()[0]  # Compute only the mean
                pl_module.log("metrics/is", inception_mean, on_epoch=True)
                self.inception_score.reset()  # Reset the metric for the next epoch

class NDB_JSD_EvaluationCallback(Callback):
    def __init__(self, number_of_bins=50, significance_level=0.05, z_threshold=None, whitening=False, max_dims=None, every_n_epochs=1):
        """
        Args:
            number_of_bins (int): Number of bins for clustering.
            significance_level (float): Significance level for hypothesis testing.
            z_threshold (float): Z-score threshold for bin comparison.
            whitening (bool): Whether to apply whitening to the samples.
            max_dims (int): Maximum dimensions to use for binning.
            every_n_epochs (int): Frequency of evaluation (in epochs).
        """
        self.n_bins = number_of_bins
        self.every_n_epochs = every_n_epochs
        self.ndb_jsd_metric = NDB_JSD_Metric(
            number_of_bins=number_of_bins,
            significance_level=significance_level,
            z_threshold=z_threshold,
            whitening=whitening,
            max_dims=max_dims
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset the NDB metric state at the start of validation."""
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.ndb_jsd_metric.reset()  # Reset the internal state before starting validation

    def on_validation_batch_end(self, trainer, pl_module, outputs,
                                 batch, batch_idx, dataloader_idx=0):
        """Update the NDB metric with new batch data."""
        if trainer.current_epoch % self.every_n_epochs == 0 and outputs is not None:
            real_images, fake_images = outputs.real_image, outputs.fake_image
            
            # Update metric with real (training) images
            self.ndb_jsd_metric.update(real_images, data_type='training')

            # Update metric with generated (fake) images
            self.ndb_jsd_metric.update(fake_images, data_type='generated')

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log NDB and JSD metrics at the end of the validation epoch."""
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Compute NDB and JSD metrics
            metrics = self.ndb_jsd_metric.compute()
            ndb_value = metrics['NDB'] / self.n_bins
            jsd_value = metrics['JS']

            # Log the metrics
            pl_module.log("metrics/ndb_k", ndb_value, on_epoch=True)
            pl_module.log("metrics/jsd", jsd_value, on_epoch=True)

# VAE-QWGAN Callback
class GMMEvaluationCallback(Callback):
    def __init__(
        self,
        gmm_components=40,
        save_gmm=False,
        gmm_save_path='gmm_model.pkl',
        ndb_jsd_args=None,
        gan_images_args=None
    ):
        """
        Args:
            every_n_epochs (int): Run the callback every N epochs.
            gmm_components (int): Number of components for the GMM.
            save_gmm (bool): Whether to save the GMM model after fitting.
            gmm_save_path (str): The file path to save the GMM model.
            ndb_jsd_args (dict): Initialization arguments for NDB_JSD_EvaluationCallback.
            gan_images_args (dict): Initialization arguments for GANImagesCallback.
        """

        self.gmm_components = gmm_components
        self.save_gmm = save_gmm
        self.gmm_save_path = gmm_save_path
        self.z_samples = []
        self.real_images = []
        
        # Initialize the existing callbacks with provided arguments
        if ndb_jsd_args is None:
            ndb_jsd_args = {}
        if gan_images_args is None:
            gan_images_args = {}
       
        self.ndb_jsd_callback = NDB_JSD_EvaluationCallback(**ndb_jsd_args)
        self.gan_images_callback = GANImagesCallback(**gan_images_args)

    def calculate_fid(self, real_imgs, fake_imgs):
        """
        Compute the Frechet Inception Distance (FID) between two sets of images.
        Assumes images are in [0, 1] and reshapes them to (N, D).
        """
        # Flatten: works for 28x28 (D=784) or any CxHxW (D = C*H*W)
        real_np = real_imgs.detach().cpu().numpy().reshape(real_imgs.shape[0], -1)
        fake_np = fake_imgs.detach().cpu().numpy().reshape(fake_imgs.shape[0], -1)
        mu1, sigma1 = real_np.mean(axis=0), np.cov(real_np, rowvar=False)
        mu2, sigma2 = fake_np.mean(axis=0), np.cov(fake_np, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid_value = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid_value
    
    def calculate_cos(self, v1, v2):
        v1 = v1.detach().cpu().numpy().reshape(-1, 784)
        v2 = v2.detach().cpu().numpy().reshape(-1, 784)
        num = np.dot(v1, np.array(v2).T) 
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1) 
        res = num / denom
        res[np.isneginf(res)] = 0
        res = 0.5 + 0.5 * res
        cos_mean = np.mean(res)
        return cos_mean
    
    def calculate_ssim(self, real_imgs, fake_imgs):
        real = real_imgs.detach().cpu().numpy().reshape(-1, 28, 28)
        fake = fake_imgs.detach().cpu().numpy().reshape(-1, 28, 28)

        ssim_values = []
        for i in range(len(real)):
            ssim_val = structural_similarity(
                real[i], 
                fake[i],
                data_range=1.0  
            )
            ssim_values.append(ssim_val)

        ssim_mean = np.mean(ssim_values)
        return ssim_mean
    
    def calculate_psnr(self, real_imgs, fake_imgs):
        real = fake_imgs.detach().cpu().numpy().reshape(-1, 28, 28)
        fake = real_imgs.detach().cpu().numpy().reshape(-1, 28, 28)

        psnr_list = []
        for i in range(len(real)):
            psnr_val = peak_signal_noise_ratio(real[i], fake[i])
            psnr_list.append(psnr_val)

        psnr_mean = np.mean(psnr_list)
        return psnr_mean
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset internal states and callbacks at the start of the validation epoch."""
        if trainer.current_epoch == trainer.max_epochs - 1:
            self.z_samples = []
            self.real_images = []
            self.ndb_jsd_callback.on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch, batch_idx):
        """Accumulate latent variables and real images during validation."""
        if trainer.current_epoch == trainer.max_epochs - 1:
            z = outputs.encode_latent
            real_images = outputs.real_image
            self.z_samples.append(z)
            self.real_images.append(real_images)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Fit GMM, generate images, and utilize existing callbacks with prefixed logging."""
        if trainer.current_epoch == trainer.max_epochs - 1:
            # Concatenate accumulated data
            real_images = torch.cat(self.real_images, dim=0)
            z_samples = torch.cat(self.z_samples, dim=0).cpu().numpy()
            
            # Fit GMM to the latent variables
            N = z_samples.shape[0]
            gmm = GaussianMixture(n_components=self.gmm_components, random_state=9).fit(z_samples)
            disp_prior, _ = gmm.sample(N)
            disp_prior = torch.tensor(disp_prior, dtype=torch.float32).to(pl_module.device)
          
            fake_images = pl_module.base_model.generator(disp_prior)

            # Save GMM model if save_gmm is True, Only save GMM on the last epoch
            if self.save_gmm and trainer.current_epoch == (trainer.max_epochs - 1):
                joblib.dump(gmm, self.gmm_save_path)
                wandb.save(self.gmm_save_path)
                print(f"GMM model saved at {self.gmm_save_path}")


            # MSE Evaluation
            val_mse_disp = F.mse_loss(real_images, fake_images, reduction='sum') / N
            pl_module.log("gmm_metrics/mse", val_mse_disp, on_epoch=True)

            fid_val = self.calculate_fid(real_images, fake_images)
            cos_sim_val = self.calculate_cos(real_images, fake_images)
            psnr_val = self.calculate_psnr(real_images, fake_images)
            ssim_val = self.calculate_ssim(real_images, fake_images)

          
            pl_module.log("gmm_metrics/fid", fid_val, on_epoch=True)
            pl_module.log("gmm_metrics/cos_sim", cos_sim_val, on_epoch=True)
            pl_module.log("gmm_metrics/psnr", psnr_val, on_epoch=True)
            pl_module.log("gmm_metrics/ssim", ssim_val, on_epoch=True)

            # Prepare outputs to mimic the structure expected by the callbacks
            outputs = ValidationResult(
                real_image=real_images,
                fake_image=fake_images,
                recon_image=None  # or set if applicable
            )

            # Wrap pl_module.log to add a prefix to metric names
            original_log_method = pl_module.log

            def prefixed_log(name, *args, **kwargs):
                prefixed_name = f"gmm_metrics/{name}"
                original_log_method(prefixed_name, *args, **kwargs)

            pl_module.log = prefixed_log

            # Wrap trainer.logger.experiment.add_image to add a prefix to image tags
            original_add_image = trainer.logger.log_image

            def prefixed_add_image(key, images,  **kwargs):
                prefixed_tag = f"gmm_metrics/{key}"
                original_add_image(key = prefixed_tag, images = images, **kwargs)

            trainer.logger.log_image = prefixed_add_image

            try:
                # Use the existing NDB/JSD callback
                self.ndb_jsd_callback.on_validation_batch_end(
                    trainer, pl_module, outputs, batch=None, batch_idx=0
                )
                self.ndb_jsd_callback.on_validation_epoch_end(trainer, pl_module)

                # Use the existing GANImagesCallback
                self.gan_images_callback.on_validation_batch_end(
                    trainer, pl_module, outputs, batch=None, batch_idx=0
                )
            finally:
                # Restore the original methods
                pl_module.log = original_log_method
                trainer.logger.log_image = original_add_image
