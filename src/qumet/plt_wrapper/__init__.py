from .qgan import (
    QGANDiscreteGenModelWrapper,
    QGANBitstringGenerationModelWrapper,
    MosaiQGANWrapper,
    PatchGANWrapper)



def get_model_wrapper(model_info, task: str):
    if model_info.is_generation_model:
        match task:
            case "discrete_generation":
                return QGANDiscreteGenModelWrapper
            case "continous_generation":
                return NotImplementedError
            case "image_generation":
                if model_info.name == 'patchgan':
                    return PatchGANWrapper
                elif model_info.name =='mosaiq':
                    return MosaiQGANWrapper
                else:
                    return ValueError(f"A training wrapper is not supported for {model_info.name}")
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
