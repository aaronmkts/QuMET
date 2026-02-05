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
    return name in VAE_MODELS


def get_vae_model_info(name: str) -> QumetModelInfo:
    if name not in VAE_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    return VAE_MODELS[name]["info"]


def get_vae_model(
    name: str,
    task: str,
    dataset_info: dict,
):
    """
    Args:
        name: The name of the model.
        task: The task type.
        dataset_info: The dataset info.

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
    raise NotImplementedError
