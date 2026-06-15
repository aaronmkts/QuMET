"""Quantum Circuit Born Machine (QCBM) models.

This module provides access to QCBM model architectures for quantum
generative modeling using Born machine approaches.
"""

from ..utils import QumetModelInfo
from .qcbm_main import get_qcbm

# fmt: off
QCBM_MODELS = {
    "qcbm": {
        "get_model_fn_generation": get_qcbm,
        "info": QumetModelInfo("qcbm", model_type="qcbm", task_type="generation", bitstring_sampling=True),
    }
}


# fmt:on
def is_qcbm_model(name: str) -> bool:
    """Check if a model name is a registered QCBM model.

    Args:
        name: Model name to check.

    Returns:
        bool: True if the model is a registered QCBM.
    """
    return name in QCBM_MODELS


def get_qcbm_model_info(name: str) -> QumetModelInfo:
    """Get metadata for a QCBM model.

    Args:
        name: Name of the QCBM model.

    Returns:
        QumetModelInfo: Model metadata and configuration.

    Raises:
        ValueError: If the model is not supported.
    """
    if name not in QCBM_MODELS:
        raise ValueError(f"QCBM model {name} is not supported")
    return QCBM_MODELS[name]["info"]


def get_qcbm_model(
    name: str,
    task: str,
    dataset_info: dict,
):
    """Get an instantiated QCBM model for a specific task.

    Args:
        name: Name of the QCBM model.
        task: Task type (e.g., 'probs_generation', 'continuous_generation').
        dataset_info: Dataset metadata for model configuration.

    Returns:
        Model instance configured for the specified task.

    Raises:
        ValueError: If the model is not supported or task is incompatible.
        AssertionError: If the model doesn't support the specified task type.
    """
    if name not in QCBM_MODELS:
        raise ValueError(f"QCBM model {name} is not supported")
    model_info: QumetModelInfo = QCBM_MODELS[name]["info"]

    match task:
        case "probs_generation":
            assert (
                model_info.bitstring_sampling
            ), f"Task {task} is not supported for {name}"
            model = QCBM_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case "continuous_generation":
            assert (
                model_info.observable_sampling
            ), f"Task {task} is not supported for {name}"
            model = QCBM_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model


def get_qcbm_model_cls(name: str):
    """Get the model class for a QCBM model.

    Args:
        name: Name of the QCBM model.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError
