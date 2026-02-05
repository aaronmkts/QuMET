from .pca_transform import PCA_Transform
from .vae_transform import VAE_Transform


def get_transform(transform_name, **kwargs):
    if transform_name == "pca":
        return PCA_Transform(**kwargs)
    elif transform_name == "vae":
        return VAE_Transform(**kwargs)
    else:
        raise ValueError(f"Unknown transform {transform_name}")
