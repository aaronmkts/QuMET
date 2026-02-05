"""Quantum Generative Adversarial Network (QGAN) models.

This module provides access to various QGAN model architectures including
EfficientSU2, PatchGAN, MosaiQ, and other quantum GAN variants.
"""

from ..utils import QumetModelInfo
from .classical_gan import get_gan
from .efficient_su2 import get_qgcd_bitstring
from .mosaiq import get_mosaiq
from .patchgan import get_patchgan
from .pwqgan import get_pqwgan_qc
from .qgan_probs import get_qgan_probs
from .qinr import get_qinr_qc
from .su2 import get_su2
from .vaeqwgan import get_vaeqwgan

# fmt: off
QGAN_MODELS = {
    "efficientsu2":{
        "get_model_fn_generation": get_qgcd_bitstring,
        "info": QumetModelInfo("efficientsu2", model_type="qgan", task_type ="generation", bitstring_sampling = True),
    },
    "qgan_probs":{
        "get_model_fn_generation": get_qgan_probs,
        "info": QumetModelInfo("qgan_probs", model_type="qgan", task_type ="generation", observable_sampling = True),
    },
    "patchgan":{
        "get_model_fn_generation": get_patchgan,
        "info": QumetModelInfo("patchgan", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "mosaiq":{
        "get_model_fn_generation": get_mosaiq,
        "info": QumetModelInfo("mosaiq", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "vaeqwgan":{
        "get_model_fn_generation": get_vaeqwgan,
        "info": QumetModelInfo("vaeqwgan", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "pqwgan_qc":{
        "get_model_fn_generation": get_pqwgan_qc,
        "info": QumetModelInfo("pqwgan_qc", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "su2":{
        "get_model_fn_generation": get_su2,
        "info": QumetModelInfo("su2", model_type="qgan", task_type ="generation",  bitstring_sampling= True),
    },
    "qinr":{
        "get_model_fn_generation": get_qinr_qc,
        "info": QumetModelInfo("qinr", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "gan":{
        "get_model_fn_generation": get_gan,
        "info": QumetModelInfo("gan", model_type="gan", task_type ="generation",  observable_sampling = True),
    },
}


# fmt:on
def is_qgan_model(name: str) -> bool:
    """Check if a model name is a registered QGAN model.
    
    Args:
        name: Model name to check.
    
    Returns:
        bool: True if the model is a registered QGAN.
    """
    return name in QGAN_MODELS


def get_qgan_model_info(name: str) -> QumetModelInfo:
    """Get metadata for a QGAN model.
    
    Args:
        name: Name of the QGAN model.
    
    Returns:
        QumetModelInfo: Model metadata and configuration.
    
    Raises:
        ValueError: If the model is not supported.
    """
    if name not in QGAN_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    return QGAN_MODELS[name]["info"]


def get_qgan_model(
    name: str,
    task: str,
    dataset_info: dict,
):
    """Get an instantiated QGAN model for a specific task.
    
    Args:
        name: Name of the QGAN model.
        task: Task type (e.g., 'probs_generation', 'image_generation').
        dataset_info: Dataset metadata for model configuration.
    
    Returns:
        Model instance configured for the specified task.
    
    Raises:
        ValueError: If the model is not supported or task is incompatible.
        AssertionError: If the model doesn't support the specified task type.
    """
    if name not in QGAN_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    model_info: QumetModelInfo = QGAN_MODELS[name]["info"]

    match task:
        case "probs_generation":
            assert (
                model_info.bitstring_sampling
            ), f"Task {task} is not supported for {name}"
            model = QGAN_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case "continuous_generation":
            assert (
                model_info.observable_sampling
            ), f"Task {task} is not supported for {name}"
            model = QGAN_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case "image_generation":
            assert (
                model_info.observable_sampling
            ), f"Task {task} is not supported for {name}"
            model = QGAN_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model


def get_qgan_model_cls(name: str):
    """Get the model class for a QGAN model.
    
    Args:
        name: Name of the QGAN model.
    
    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError
