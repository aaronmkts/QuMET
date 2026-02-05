"""Transform functions for vision datasets.

This module provides transformation pipelines for vision datasets including
data augmentation, normalization, PCA, and other preprocessing operations.
"""

import numpy as np
import torch

from .cifar import get_cifar10_transform
from .fashion_mnist import get_fashion_mnist_transform
from .mnist import get_mnist_transform
from .utils import PCA, filter_by_labels, n_samples, scale_data


class DataTransformationMixin:
    """Mixin class providing common data transformation methods.
    
    Provides utilities for filtering labels, sampling, and applying PCA
    transformations to dataset instances.
    """

    def filter_labels(self, labels: list):
        """Filter the dataset to only include specified labels.
        
        Args:
            labels: List of label values to keep.
        
        Returns:
            Filtered dataset.
        """
        return filter_by_labels(self, labels)

    def n_samples_(self, samples: int):
        """Limit the dataset to a specified number of samples.
        
        Args:
            samples: Number of samples to keep.
        
        Returns:
            Dataset with limited samples.
        """

        return n_samples(self, samples)

    def fit_pca(self, n_components):
        """Fit PCA transformation to the dataset.
        
        Args:
            n_components: Number of principal components to keep.
        """

        self.pca_model = PCA(n_components)

        X_flat_data = self.data.numpy().reshape(self.data.shape[0], -1)
        X_data = self.scale_data(X_flat_data, scale=[0, 1])
        self.pca_model.fit(torch.tensor(X_data, dtype=torch.float32))

    def apply_pca(self, img):
        """Apply fitted PCA transformation to an image.
        
        Args:
            img: Input image tensor.
        
        Returns:
            torch.Tensor: PCA-transformed image.
        """

        transformed_img = self.pca_model.transform(img)

        return transformed_img

    @staticmethod
    def scale_data(data, scale=None, dtype=np.float32):
        """Scale data to a specified range.
        
        Args:
            data: Input data to scale.
            scale: Target scale range [min, max]. Defaults to None.
            dtype: Output data type. Defaults to np.float32.
        
        Returns:
            Scaled data array.
        """
        return scale_data(data, scale=scale, dtype=np.float32)


def get_vision_dataset_transform(
    name: str, train: bool, model_name: str, transform: None
):
    """Get transformation pipeline for a vision dataset.
    
    Args:
        name: Name of the vision dataset.
        train: Whether this is for training (affects augmentation).
        model_name: Optional model name for model-dependent transforms.
        transform: Pre-configured transform (if callable) or transform type string.
    
    Returns:
        Callable transform function or transform instance.
    
    Raises:
        ValueError: If dataset name is not recognized.
    """
    if callable(transform):
        return transform
    else:
        match name.lower():
            case "mnist":
                return get_mnist_transform(train, model_name)
            case "fashion_mnist":
                return get_fashion_mnist_transform(train, model_name)
            case "cifar10":
                return get_cifar10_transform(train, model_name)
            case _:
                raise ValueError(f"Unknown dataset {name}")
