from ..utils import QumetModelInfo

from .efficient_su2 import get_qgcd_bitstring
from .patchgan import get_patchgan
from .qgcd_probs import get_qgcd_probs
from .mosaiq import get_mosaiq
from .apqgan import get_apqgan
from .pwqgan import get_pqwgan_qc

# fmt: off
QGAN_MODELS = {
    "efficientsu2":{
        "get_model_fn_generation": get_qgcd_bitstring,
        "info": QumetModelInfo("efficientsu2", model_type="qgan", task_type ="generation", bitstring_sampling = True),
    },
    "qgcd_probs":{
        "get_model_fn_generation": get_qgcd_probs,
        "info": QumetModelInfo("qgcd_probs", model_type="qgan", task_type ="generation", observable_sampling = True),
    },
    "patchgan":{
        "get_model_fn_generation": get_patchgan,
        "info": QumetModelInfo("patchgan", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "mosaiq":{
        "get_model_fn_generation": get_mosaiq,
        "info": QumetModelInfo("mosaiq", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "sspqgan":{
        "get_model_fn_generation": get_apqgan,
        "info": QumetModelInfo("sspqgan", model_type="qgan", task_type ="generation",  observable_sampling = True),
    },
    "pqwgan_qc":{
        "get_model_fn_generation": get_pqwgan_qc,
        "info": QumetModelInfo("pqwgan_qc", model_type="qgan", task_type ="generation",  observable_sampling = True),
    }

}


# fmt:on
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
        case "discrete_generation":
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
    raise NotImplementedError
