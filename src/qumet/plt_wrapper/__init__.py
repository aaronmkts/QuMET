from .qgan import (
    QGANDiscreteGenModelWrapper,
    QGANBitstringGenerationModelWrapper,
    MosaiQGANWrapper,
    PatchGANWrapper,
    SSPQGANWrapper)

from .qcbm import(
QCBMProbsGenModelWrapper
)
from .vae import VAEWrapper


def get_model_wrapper(model_info, task: str):

    if model_info.is_generation_model:
        
        if model_info.model_type.value == 'qgan':
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
                    elif model_info.name == 'sspqgan':
                        return SSPQGANWrapper
                    else:
                        return ValueError(f"A training wrapper is not supported for {model_info.name}")
                case _:
                    raise ValueError(f"Task {task} is not supported for {model_info.name}")

        elif model_info.model_type.value == 'qcbm':
            match task:
                case 'probs_generation':
                    return QCBMProbsGenModelWrapper
                case _:
                    raise ValueError(f"Task {task} is not supported for {model_info.name}")

        elif model_info.model_type.value == 'vae':
            match task:
                case "image_generation":
                    if model_info.name == 'vae':    
                        return VAEWrapper
                    else:
                        return ValueError(f"A training wrapper is not supported for {model_info.name}")
                case _:
                    raise ValueError(f"Task {task} is not supported for {model_info.name}")
                
