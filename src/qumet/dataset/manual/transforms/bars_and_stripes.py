def _get_bars_and_stripes_default_transform(transform: str, discretise: bool):

    normaliser, discretisation = None, None
    return normaliser, discretisation


def get_bars_and_stripes_default_transform(transform: str, discretise: bool):
    return _get_bars_and_stripes_default_transform(transform, discretise)


def get_bars_and_stripes_transform(transform: str, discretise: bool, model: str = None):
    if model is None:
        return get_bars_and_stripes_default_transform(transform, discretise)
    else:
        # Currently no model-dependent transform for 2d_gaussian is supported.
        return get_bars_and_stripes_default_transform(transform, discretise)
