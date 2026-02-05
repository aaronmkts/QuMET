"""Model Zoo for QuMET.

This module provides a unified interface for accessing and instantiating various
quantum machine learning models including QGANs, QCBMs, and VAEs.
"""

from os import PathLike

from .qcbm import (
    get_qcbm_model,
    get_qcbm_model_info,
    is_qcbm_model,
)
from .qgan import (
    get_qgan_model,
    get_qgan_model_info,
    is_qgan_model,
)
from .utils import ModelType, QumetModelInfo
from .vae import (
    get_vae_model,
    get_vae_model_info,
    is_vae_model,
)


def get_model_info(name: str) -> QumetModelInfo:
    """Retrieve model metadata and configuration.
    
    Args:
        name: Name of the model to query.
    
    Returns:
        QumetModelInfo: Model metadata including type and capabilities.
    
    Raises:
        ValueError: If the model name is not recognized.
    """
    if is_qgan_model(name):
        info = get_qgan_model_info(name)

    elif is_qcbm_model(name):
        info = get_qcbm_model_info(name)

    elif is_vae_model(name):
        info = get_vae_model_info(name)

    else:
        raise ValueError(f"Model {name} not found")

    return info


def get_model(
    name: str,
    task: str,
    dataset_info: dict = {},
):
    """Get an instantiated model by name and task.
    
    Args:
        name: Name of the model to instantiate.
        task: Task type for the model (e.g., 'discrete_generation').
        dataset_info: Optional dataset metadata for model configuration.
    
    Returns:
        Model instance configured for the specified task.
    
    Raises:
        ValueError: If the model type is not supported.
    """
    model_info = get_model_info(name)

    model_kwargs = {
        "name": name,
        "task": task,
        "dataset_info": dataset_info,
    }

    match model_info.model_type:
        case ModelType.QGAN | ModelType.GAN:
            model = get_qgan_model(**model_kwargs)
        case ModelType.QCBM:
            model = get_qcbm_model(**model_kwargs)
        case ModelType.VAE:
            model = get_vae_model(**model_kwargs)
        case _:
            raise ValueError(f"Model type {model_info.model_type} not supported")
    return model
