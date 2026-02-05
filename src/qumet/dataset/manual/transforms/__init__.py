"""Transform functions for manual quantum datasets.

This module provides transformation pipelines for manual datasets including
normalization, discretization, and dataset-specific transformations.
"""

from .bars_and_stripes import get_bars_and_stripes_transform
from .two_d_gaussian import get_two_d_gaussian_transform
from .two_d_grid_gaussian import get_two_d_grid_gaussian_transform
from .two_d_ring_gaussian import get_two_d_ring_gaussian_transform


def get_manual_dataset_transform(
    name: str, transform: str, discretise: bool, model_name: str
):
    """Get transformation pipeline for a manual dataset.

    Args:
        name: Name of the manual dataset.
        transform: Transform type to apply.
        discretise: Whether to discretise the data.
        model_name: Optional model name for model-dependent transforms.

    Returns:
        tuple: (normaliser, discretisation) transform functions.

    Raises:
        ValueError: If dataset name is not recognized.
    """
    match name.lower():
        case "bars_and_stripes":
            return get_bars_and_stripes_transform(transform, discretise, model_name)
        case "2d_gaussian":
            return get_two_d_gaussian_transform(transform, discretise, model_name)
        case "2d_grid_gaussian":
            return get_two_d_grid_gaussian_transform(transform, discretise, model_name)
        case "2d_ring_gaussian":
            return get_two_d_ring_gaussian_transform(transform, discretise, model_name)
        case _:
            raise ValueError(f"Unknown dataset {name}")
