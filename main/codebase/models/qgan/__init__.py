from ..utils import QumetModelInfo
from os import PathLike

from .qgcd import QGCDConfig, Generator


QGAN_MODELS = {
    "qgcd":{
        "config_cls": QGCDConfig,
        "info": QumetModelInfo("qgcd", model_type="qgan", task_type ="generation", data_generation = True),
        "continuous_modelling": Generator
    }
}

def is_qgan_model(name: str) -> bool:
    return name in QGAN_MODELS

def get_qgan_model_info(name: str) -> QumetModelInfo:
    if name not in QGAN_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    return QGAN_MODELS[name]["info"]

def get_qgan_model_config(name: str) -> type:
    if name not in QGAN_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    return QGAN_MODELS[name]["config_cls"]

def get_qgan_model(
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
    if name not in QGAN_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    model_info: QumetModelInfo = QGAN_MODELS[name]["info"]

    if task in ["generation"]:
        assert (
            model_info.data_generation
        ), f"Task {task} is not supported for {name}"

        config_cls = QGAN_MODELS[name]["config_cls"]
        model_cls = QGAN_MODELS[name]["continuous_modelling"]
        
    else:
        raise ValueError(f"Task {task} is not supported for {name}")
    
    model = model_cls(config_cls)

    return model