from .qgan import QGANGenerationModelWrapper


def get_model_wrapper(model_info, task: str):
    if model_info.is_generation_model:
        return QGANGenerationModelWrapper
    else:
        raise ValueError(f"Task {task} is not supported for {model_info.name}")
