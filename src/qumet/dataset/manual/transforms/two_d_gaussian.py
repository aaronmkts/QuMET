from .utils import MinMaxNormalizer, PITNormalizer, compute_discretization


def _get_two_d_gaussian_default_transform(transform: str, discretise: bool):
    epsilon = 1e-6

    match transform.lower():
        case "minmax":
            normaliser = MinMaxNormalizer(epsilon=epsilon)
        case "pit":
            normaliser = PITNormalizer(epsilon=epsilon)
        case _:
            raise ValueError("Transformation value must be either 'minmax' or 'pit")

    match discretise:
        case True:
            discretisation = compute_discretization
        case False:
            discretisation = None

    return normaliser, discretisation


def get_two_d_gaussian_default_transform(transform: str, discretise: bool):
    return _get_two_d_gaussian_default_transform(transform, discretise)


def get_two_d_gaussian_transform(transform: str, discretise: bool, model: str = None):

    if model is None:
        return get_two_d_gaussian_default_transform(transform, discretise)
    else:
        # Currently no model-dependent transform for 2d_gaussian is supported.
        return get_two_d_gaussian_default_transform(transform, discretise)
