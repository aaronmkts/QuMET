"""Variational Autoencoder (VAE) models.

This module provides access to VAE model architectures for classical
generative modeling and image generation tasks.
"""

from os import PathLike

from ..utils import QumetModelInfo
from .vae import get_vae

# fmt: off

VAE_MODELS = {
    "vae":{
        "get_model_fn_generation": get_vae,
        "info": QumetModelInfo("vae", model_type="vae", task_type ="generation", observable_sampling = True),
    }
}


# fmt:on
def is_vae_model(name: str) -> bool:
    """Check if a model name is a registered VAE model.

    Args:
        name: Model name to check.

    Returns:
        bool: True if the model is a registered VAE.
    """
    return name in VAE_MODELS


def get_vae_model_info(name: str) -> QumetModelInfo:
    """Get metadata for a VAE model.

    Args:
        name: Name of the VAE model.

    Returns:
        QumetModelInfo: Model metadata and configuration.

    Raises:
        ValueError: If the model is not supported.
    """
    if name not in VAE_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    return VAE_MODELS[name]["info"]


def get_vae_model(
    name: str,
    task: str,
    dataset_info: dict,
):
    """Get an instantiated VAE model for a specific task.

    Args:
        name: Name of the VAE model.
        task: Task type (e.g., 'image_generation').
        dataset_info: Dataset metadata for model configuration.

    Returns:
        Model instance configured for the specified task.

    Raises:
        ValueError: If the model is not supported or task is incompatible.
        AssertionError: If the model doesn't support the specified task type.
    """
    if name not in VAE_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    model_info: QumetModelInfo = VAE_MODELS[name]["info"]

    match task:

        case "image_generation":
            assert (
                model_info.observable_sampling
            ), f"Task {task} is not supported for {name}"
            model = VAE_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model


def get_qgan_model_cls(name: str):
    """Get the model class for a VAE model.

    Args:
        name: Name of the VAE model.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError
