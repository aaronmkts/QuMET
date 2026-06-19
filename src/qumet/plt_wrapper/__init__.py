from .qgan import (
    QGANDiscreteGenModelWrapper,
    QGANGenerationModelWrapper,
    MosaiQGANWrapper,
    PatchGANWrapper,
    APQGANWrapper,
    PQWGANWrapper,
    ProbsQGANWrapper,
    QINRWrapper,
    GANWrapper
    )

from .qcbm import(
QCBMDiscreteGenModelWrapper
)
from .vae import VAEWrapper
from .vaeqgan_wrapper import VAEGANWrapper


def get_model_wrapper(model_info, task: str, add_vae: bool = False):

    if add_vae:
        if model_info.model_type.value in ('qgan', 'gan') and task == "image_generation":
            return VAEGANWrapper
        elif model_info.model_type.value in ('qgan', 'gan') and task != "image_generation":
            raise ValueError(f"VAE-GAN wrapper is not supported for task {task} in model {model_info.name}")
        else:
            raise ValueError(f"VAE-GAN wrapper is not supported for model type {model_info.model_type.value}")
    
    if model_info.is_generation_model:
        if model_info.model_type.value =='gan':
            match task:
                case "image_generation":
                    if model_info.name == 'gan':
                        return GANWrapper
                    else:
                        return ValueError(f"A training wrapper is not supported for {model_info.name}")
                case _:
                    raise ValueError(f"Task {task} is not supported for {model_info.name}")
                
        if model_info.model_type.value == 'qgan':
            match task:
                case "discrete_generation":
                    return QGANDiscreteGenModelWrapper
                case "continuous_generation":
                    return QGANGenerationModelWrapper
                case "image_generation":
                    if model_info.name == 'patchgan':
                        return PatchGANWrapper
                    elif model_info.name =='mosaiq':
                        return MosaiQGANWrapper
                    elif model_info.name == 'apqgan':
                        return APQGANWrapper
                    elif model_info.name == 'pqwgan_qc':
                        return PQWGANWrapper
                    elif model_info.name == 'qinr':
                        return QINRWrapper
                    else:
                        return ValueError(f"A training wrapper is not supported for {model_info.name}")
                case _:
                    raise ValueError(f"Task {task} is not supported for {model_info.name}")

        elif model_info.model_type.value == 'qcbm':
            match task:
                case 'discrete_generation':
                    return QCBMDiscreteGenModelWrapper
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
                
