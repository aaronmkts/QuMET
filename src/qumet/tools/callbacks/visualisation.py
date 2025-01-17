
from lightning import LightningModule, Trainer
import numpy as np
from lightning.pytorch.callbacks import Callback
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from qumet.plt_wrapper.base import ValidationResult
from sklearn.manifold import TSNE
import torch 
import io
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.colors as mcolors
import os 

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def get_grid_images(imgs, nimgs=64, nrow=8):
    """Create a grid of images for visualization."""
    grid = torchvision.utils.make_grid(denorm(imgs[:nimgs]), normalize=False, nrow=nrow, pad_value=1)
    return grid
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


class TSNEPlotCallback(Callback):
    def __init__(self,
                 every_n_epochs: int = 1,
                 random_state: int = 42,
                 flatten: bool = True):

        self.every_n_epochs = every_n_epochs
        self.random_state = random_state
        self.flatten = flatten

        # Buffers
        self.real_images = []
        self.real_labels = []
        self.fake_images = []
        self.fake_labels = []

        # Will store the t-SNE embeddings
        self.real_tsne = None
        self.real_labels_saved = None
        self.unique_labels = None

    def on_validation_batch_end(self, trainer, pl_module, outputs: ValidationResult, batch_idx, dataloader_idx=0):

        if outputs is not None and hasattr(outputs, "real_image") and hasattr(outputs, "fake_image"):
            real_imgs = outputs.real_image
            fake_imgs = outputs.fake_image

            real_labels = getattr(outputs, "label", None)

            # We only collect real images on epoch == 0 (or the first time we do this),
            # or whenever we haven't yet computed the real dataset's t-SNE.
            if trainer.current_epoch == 0 and self.real_tsne is None:
                self._collect_samples(self.real_images, real_imgs)
                if real_labels is not None:
                    self._collect_samples(self.real_labels, real_labels)

            # We collect fake images if we are going to do a t-SNE plot this epoch.
            if trainer.current_epoch == 0 or (trainer.current_epoch % self.every_n_epochs == 0):
                self._collect_samples(self.fake_images, fake_imgs)

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.current_epoch == 0 and self.real_tsne is None:
            # Compute real images t-SNE
            if len(self.real_images) > 0:
                real_data = torch.cat(self.real_images, dim=0)
                real_labels = None

                if len(self.real_labels) > 0:
                    real_labels = torch.cat(self.real_labels, dim=0)


                if self.flatten:
                    real_data = real_data.view(real_data.size(0), -1)

                self.real_tsne = self._compute_tsne(real_data)
               
                self.real_labels_saved = real_labels.cpu().numpy() if real_labels is not None else None
                self.unique_labels = np.unique(self.real_labels_saved)

            # Clear buffers for real images (we won't need them again)
            self.real_images.clear()
            self.real_labels.clear()

 
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Only proceed if we actually have any fake images
            if len(self.fake_images) > 0:
                fake_data = torch.cat(self.fake_images, dim=0)
                fake_labels = None

                # Flatten if needed
                if self.flatten:
                    fake_data = fake_data.view(fake_data.size(0), -1)

                fake_tsne = self._compute_tsne(fake_data)

                # Plot
                self._plot_tsne(
                    trainer,
                    real_tsne=self.real_tsne,
                    real_labels=self.real_labels_saved,
                    fake_tsne=fake_tsne,
                    epoch=trainer.current_epoch
                )

            # Clear fake buffers after each epoch
            self.fake_images.clear()
            self.fake_labels.clear()

    def _compute_tsne(self, data: torch.Tensor) -> np.ndarray:
        """
        Compute the t-SNE embedding for the given data (Tensor -> np.ndarray).
        """
        data_np = data.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=self.random_state)
        emb = tsne.fit_transform(data_np)
        return emb
    
    def _log_image(self, trainer, img, name):
        trainer.logger.experiment.add_image(name, img, global_step=trainer.current_epoch)


    def _plot_tsne(self,
                   trainer,
                   real_tsne: np.ndarray,
                   real_labels: np.ndarray,
                   fake_tsne: np.ndarray,

                   epoch: int):
        """
        Create a scatter plot of real and fake images in TSNE space.
        - real_tsne, fake_tsne: (N, 2) arrays
        - real_labels, fake_labels: (N,) arrays or None
        """
        plt.figure(figsize=(10, 8))

        if real_tsne is not None:
            if real_labels is not None:
                n_labels = len(self.unique_labels)
                label_mapping = {old: new for new, old in enumerate(self.unique_labels)}
                real_labels_mapped = np.array([label_mapping[label] for label in real_labels])

                scatter = plt.scatter(
                    real_tsne[:, 0],
                    real_tsne[:, 1],
                    c=real_labels_mapped,
                    cmap=plt.cm.get_cmap("tab10", n_labels),
                    alpha=0.7,
                    label="Real Data"
                )
                cbar = plt.colorbar(scatter)
                cbar.set_ticks(range(n_labels))
                cbar.set_ticklabels(self.unique_labels)
            else:
                # Otherwise, plot real data as one color
                plt.scatter(real_tsne[:, 0], real_tsne[:, 1],
                            c='blue', alpha=0.5, label="Real Data")

            if fake_tsne is not None:
                # Plot fake data on top 
                plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1],
                            c='black', alpha=0.75, marker='o', label="Fake Data")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")  
        buf.seek(0)
        
        plot_img = Image.open(buf)
        transform = transforms.ToTensor()
        plot_tensor = transform(plot_img)

        self._log_image(
            trainer=trainer,     
            img=plot_tensor,
            name=f"TSNE/epoch_{epoch}"  
        )
        base_dir = os.path.abspath(os.path.join(trainer.log_dir, "../../"))
        save_file = os.path.join(base_dir, f"tsne_epoch_{epoch}.png")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file)


        plt.close()
        buf.close()

    def _collect_samples(self, storage_list, new_data):
   
        if new_data is None:
            return
        # Ensure it's detached from graph
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.detach().cpu()
        storage_list.append(new_data)















































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