# Model Zoo for QuMET
from os import PathLike

from .qgan import (
    is_qgan_model,
    get_qgan_model,
    get_qgan_model_info,
)
from .vae import (
    is_vae_model,
    get_vae_model,
    get_vae_model_info,
)

from .qcbm import(
    is_qcbm_model,
    get_qcbm_model,
    get_qcbm_model_info,
)
from .utils import QumetModelInfo, ModelType


def get_model_info(name: str) -> QumetModelInfo:
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
    dataset_info: dict,
):
    model_info = get_model_info(name)

    model_kwargs = {
        "name": name,
        "task": task,
        "dataset_info": dataset_info,
    }

    match model_info.model_type:
        case ModelType.QGAN:
            model = get_qgan_model(**model_kwargs)
        case ModelType.QCBM:
            model = get_qcbm_model(**model_kwargs)
        case ModelType.VAE:
            model = get_vae_model(**model_kwargs)
        case _:
            raise ValueError(f"Model type {model_info.model_type} not supported")
    return model
