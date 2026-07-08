"""Generate bars-and-stripes bitstring patterns."""

import itertools

import numpy as np


def bars_and_stripes(rows: int, cols: int) -> np.ndarray:
    """Return the unique bars-and-stripes patterns for a grid."""
    data = []

    for row_pattern in itertools.product([0, 1], repeat=cols):
        picture = np.repeat([row_pattern], rows, axis=0)
        data.append(picture.ravel().tolist())

    for col_pattern in itertools.product([0, 1], repeat=rows):
        picture = np.repeat(np.array(col_pattern).reshape(rows, 1), cols, axis=1)
        data.append(picture.ravel().tolist())

    return np.unique(np.asarray(data, dtype=int), axis=0)
