from ..utils import QumetModelInfo
from os import PathLike

from .qgcd import QGCDConfig, Generator
from .qgcd_probs import get_qgcd_probs

#fmt: off
QGAN_MODELS = {
    "qgcd":{
        "config_cls": QGCDConfig,
        "info": QumetModelInfo("qgcd", model_type="qgan", task_type ="generation", data_generation = True),
        "continuous_modelling": Generator
    },
    "qgcd_probs":{
        "get_model_fn_generation": get_qgcd_probs,
        "info": QumetModelInfo("qgcd_probs", model_type="qgan", task_type ="generation", data_generation = True),
    }
}

#fmt:on
def is_qgan_model(name: str) -> bool:
    return name in QGAN_MODELS

def get_qgan_model_info(name: str) -> QumetModelInfo:
    if name not in QGAN_MODELS:
        raise ValueError(f"QGAN model {name} is not supported")
    return QGAN_MODELS[name]["info"]

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

    match task:
        case "generation":
            assert(
                model_info.data_generation
            ), f"Task {task} is not supported for {name}"
            model = QGAN_MODELS[name]["get_model_fn_generation"](
                info = dataset_info
            )

            
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model

def get_qgan_model_cls(name: str):
    raise NotImplementedError