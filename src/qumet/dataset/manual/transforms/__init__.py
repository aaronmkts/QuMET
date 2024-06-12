from .two_d_gaussian import get_two_d_gaussian_transform
from .two_d_grid_gaussian import get_two_d_grid_gaussian_transform
from .two_d_ring_gaussian import get_two_d_ring_gaussian_transform


def get_manual_dataset_transform(name: str, transform: str, discretise: bool, model_name: str):
    """
    Args:
        name (str): name of the dataset
        train (bool): whether the dataset is used for training
        model_name (Optional[str, None]): name of the model. Some pretrained models have model-dependent transforms.
    Returns:
        transform (callable): transform function
    """
    match name.lower():
        case "2d_gaussian":
            return get_two_d_gaussian_transform(transform, discretise, model_name)
        case "2d_grid_gaussian":
            return get_two_d_grid_gaussian_transform(transform, discretise, model_name)
        case "2d_ring_gaussian":
            return get_two_d_ring_gaussian_transform(transform, discretise, model_name)
        case _:
            raise ValueError(f"Unknown dataset {name}")
