"""Advanced transformation methods for vision datasets.

This module provides advanced transformations including PCA and VAE-based
dimensionality reduction for vision datasets.
"""

from .pca_transform import PCA_Transform
from .vae_transform import VAE_Transform


def get_transform(transform_name, **kwargs):
    """Get a transformation instance by name.
    
    Args:
        transform_name: Name of the transform ('pca' or 'vae').
        **kwargs: Additional arguments to pass to the transform constructor.
    
    Returns:
        Transform instance.
    
    Raises:
        ValueError: If transform name is not recognized.
    """
    if transform_name == "pca":
        return PCA_Transform(**kwargs)
    elif transform_name == "vae":
        return VAE_Transform(**kwargs)
    else:
        raise ValueError(f"Unknown transform {transform_name}")
