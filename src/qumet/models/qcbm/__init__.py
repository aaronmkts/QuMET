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
    return name in QCBM_MODELS


def get_qcbm_model_info(name: str) -> QumetModelInfo:
    if name not in QCBM_MODELS:
        raise ValueError(f"QCBM model {name} is not supported")
    return QCBM_MODELS[name]["info"]


def get_qcbm_model(
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
    raise NotImplementedError
