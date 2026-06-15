import torch
from torchvision import transforms as tv_transforms

from ..utils import PCA
from .base_transform import TransformBase


class PCA_Transform(TransformBase):
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca_model = PCA(n_components=self.n_components)

    def fit(self, data):
        # data is expected to be a PyTorch tensor
        self.pca_model.fit(data)

    def __call__(self, x):
        # x can be a PIL Image or a PyTorch tensor
        x_tensor = x if isinstance(x, torch.Tensor) else tv_transforms.ToTensor()(x)
        x_flat = x_tensor.view(-1)
        x_transformed = self.pca_model.transform(
            x_flat.unsqueeze(0)
        )  # Add batch dimension
        return x_transformed.squeeze(0)  # Remove batch dimension

    def inverse_transform(self, x, img_shape=(1, 28, 28)):
        # Ensure x is a 2D tensor (batch_size, n_components)

        if x.ndimension() != 2:
            raise ValueError(
                f"Expected input tensor to have 2 dimensions (batch_size, n_components), but got shape {x.shape}"
            )

        x_reconstructed = self.pca_model.inverse_transform(x)
        batch_size = x_reconstructed.shape[0]

        return x_reconstructed.view(batch_size, *img_shape)
