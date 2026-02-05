import torch
from torchvision import transforms as tv_transforms

from .base_transform import TransformBase


class VAE_Transform(TransformBase):
    def __init__(self, vae_model):
        self.vae_model = vae_model

    def fit(self, data):
        # Train the VAE model here if not already trained
        # self.vae_model.train()
        # Training code goes here
        pass

    def __call__(self, x):
        x_tensor = x if isinstance(x, torch.Tensor) else tv_transforms.ToTensor()(x)
        x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            z, _, _ = self.vae_model.encode(x_tensor)
        return z.squeeze(0)

    def inverse_transform(self, z):
        z = z.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            x_reconstructed = self.vae_model.decode(z)
        return x_reconstructed.squeeze(0)
