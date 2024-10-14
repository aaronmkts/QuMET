from ..utils import QumetModelInfo
from .gap import get_gap


# fmt: off
GAP_MODELS = {
    "gap":{
        "get_model_fn_generation": get_gap,
        "info": QumetModelInfo("gap", model_type="gap", task_type ="generation", observable_sampling = True),
    },
}


# fmt:on
def is_gap_model(name: str) -> bool:
    return name in GAP_MODELS


def get_gap_model_info(name: str) -> QumetModelInfo:
    if name not in GAP_MODELS:
        raise ValueError(f"GAP model {name} is not supported")
    return GAP_MODELS[name]["info"]


def get_gap_model(
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
    if name not in GAP_MODELS:
        raise ValueError(f"GAP model {name} is not supported")
    model_info: QumetModelInfo = GAP_MODELS[name]["info"]
    
    match task:
        case "probs_generation":
            assert (
                model_info.bitstring_sampling
            ), f"Task {task} is not supported for {name}"
            model = GAP_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case "continuous_generation":
            assert (
                model_info.observable_sampling
            ), f"Task {task} is not supported for {name}"
            model = GAP_MODELS[name]["get_model_fn_generation"](info=dataset_info)
        
        case "image_generation":
            assert (
                model_info.observable_sampling
            ), f"Task {task} is not supported for {name}"
            model = GAP_MODELS[name]["get_model_fn_generation"](info=dataset_info)

        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model


def get_gap_model_cls(name: str):
    raise NotImplementedError
