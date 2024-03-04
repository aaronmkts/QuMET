from .qgan import (
    QGANGenerationModelWrapper,
    QGANBitstringGenerationModelWrapper
)



def get_model_wrapper(model_info, task: str):
    if model_info.is_generation_model:
        match task:
            case "probs_generation":
                return QGANGenerationModelWrapper
            case "bitsring_generation":
                return QGANBitstringGenerationModelWrapper
            case "image_generation":
                return NotImplementedError
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
