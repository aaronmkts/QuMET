from ..utils import QumetModelInfo
from os import PathLike

from .qgan import QganConfig

'''
MANUAL_MODELS = {
    "qgan":{
        "config": QganConfig,
        "info": QumetModelInfo("qgan", model_source="manual", task_type ="vision")
    }
}

def is_manual_model(name: str) -> bool:
    return name in MANUAL_MODELS

def get_manual_model_info(name: str) -> QumetModelInfo:
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    return MANUAL_MODELS[name]["info"]

def get_manual_model_config(name: str) -> type:
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    return MANUAL_MODELS[name]["config"]

def get_manual_model(
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
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    model_info: QumetModelInfo = MANUAL_MODELS[name]["info"]

    if task in ["generation"]:
        assert (
            model_info.image_classification
        ), f"Task {task} is not supported for {name}"

        config = MANUAL_MODELS[name]["config"]
        model_ = MANUAL_MODELS[name]["vision"]
        
    else:
        raise ValueError(f"Task {task} is not supported for {name}")
    
    model = model_(config)

    return model'''
