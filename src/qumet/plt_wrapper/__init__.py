from .qgan import (
    QGANDiscreteGenModelWrapper,
    QGANBitstringGenerationModelWrapper,
    QGANImageGenerationModelWrapper)



def get_model_wrapper(model_info, task: str):
    if model_info.is_generation_model:
        match task:
            case "discrete_generation":
                return QGANDiscreteGenModelWrapper
            case "continous_generation":
                return NotImplementedError
            case "image_generation":
                return QGANImageGenerationModelWrapper
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
