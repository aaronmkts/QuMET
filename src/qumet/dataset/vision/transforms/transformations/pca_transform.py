import torch
from torchvision import transforms as tv_transforms
from .base_transform import TransformBase
from ..utils import PCA

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
        x_transformed = self.pca_model.transform(x_flat.unsqueeze(0))  # Add batch dimension
        return x_transformed.squeeze(0)  # Remove batch dimension

    def inverse_transform(self, x):
        # x is a tensor of PCA components
        x_reconstructed = self.pca_model.inverse_transform(x.unsqueeze(0))  # Add batch dimension
        x_reconstructed = x_reconstructed.squeeze(0)  # Remove batch dimension
        # Reshape to original image dimensions if necessary
        # For MNIST, reshape to (1, 28, 28)
        return x_reconstructed.view(1, 28, 28)
