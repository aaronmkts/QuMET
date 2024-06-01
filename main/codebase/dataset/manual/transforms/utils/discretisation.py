from itertools import product
import numpy as np


def center_2d(i, j, n):
    ax, bx = 0., 1.
    ay, by = 0., 1.
    return ax + (2 * i + 1) / (2 * n) * (bx - ax), ay + (2 * j + 1) / (2 * n) * (by - ay)


def center(coord, n):
    return np.array(coord) / n + 0.5 / n


def compute_discretization(n_qubits, n_dim):
    format_string = "{:0" + str(n_qubits) + "b}"
    n = 2 ** (n_qubits // n_dim)
    dict_bins = {}

    for k, coordinates in enumerate(product(range(n), repeat=n_dim)):
        dict_bins.update({
            format_string.format(k): [coordinates, center(coordinates, n)]
        })
    return dict_bins
