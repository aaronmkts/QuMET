# Model Zoo for QuMET
from os import PathLike
'''
from .manual import (
    is_manual_model,
    get_manual_model,
    get_manual_model_config,
    get_manual_model_info,
)

'''
from .utils import QumetModelInfo, ModelSource, ModelTaskType

"""

def get_model_info(name: str) -> QumetModelInfo:
    if is_manual_model(name):
        info = get_manual_model_info(name)
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


    match model_info.model_source:
        case ModelSource.MANUAL:
            model = get_manual_model(**model_kwargs)
        case _:
            raise ValueError(f"Model source {model_info.model_source} not supported")
    return model"""
