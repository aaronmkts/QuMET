from .configuration_qgan import QganConfig

f = QganConfig
"""
def get_qgan(
    name: str,
    task: str,
    info: dict,
):
    
    if task not in ["generation"]:
        raise ValueError(f"Task {task} is not supported for qgan")

    match task:
        case "generation":

            config = QganConfig
            model = Generator

        case _:
            raise ValueError(f"Task {task} is not supported for qgan")

    return model
"""
