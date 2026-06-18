from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from .integral_transform import emp_integral_trans
from typing import List

# Transformation classes

class MinMaxNormalizer:
    def __init__(self, reverse_lookup = None, epsilon = 0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.min = data.min()
        self.max = data.max() - data.min()
        data = (data - self.min) / self.max
        self.reverse_lookup = (self.min, self.max)
        return data / (1 + self.epsilon)

    def transform(self, data: np.ndarray) -> np.ndarray:
        min = data.min()
        max = data.max() - data.min()
        data = (data - min) / max
        return data / (1 + self.epsilon)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data * (1 + self.epsilon)
        self.min, self.max = self.reverse_lookup
        return data * self.max + self.min


class PITNormalizer():
    def __init__(self, reverse_lookup = None, epsilon = 0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        epit = np.array(data, copy=True).T
        reverse_epit_lookup = np.array(epit, copy=True)

        epit = np.array([emp_integral_trans(row) for row in epit]).T
        reverse_epit_lookup = np.array([np.sort(row) for row in reverse_epit_lookup])

        self.reverse_lookup = jnp.array(reverse_epit_lookup)
        return epit / (1 + self.epsilon)

    def transform(self, data: np.ndarray) -> np.ndarray:
        epit = np.array(data, copy=True).T
        epit = np.array([emp_integral_trans(row) for row in epit]).T
        return epit / (1 + self.epsilon)

    def _reverse_emp_integral_trans_single(self, values: jnp.ndarray) -> List[float]:
    # assumes non ragged array
        values = values * (jnp.shape(self.reverse_lookup)[1] - 1)
        rows = jnp.shape(self.reverse_lookup)[0]
    # if we are an integer do not use linear interpolation
        valuesL = jnp.floor(values).astype(int)
        valuesH = jnp.ceil(values).astype(int)
    # if we are an integer then floor and ceiling are the same
        isIntMask = 1 - (valuesH - valuesL)
        rowIndexer = jnp.arange(rows)
        resultL = self.reverse_lookup[([rowIndexer], [valuesL])]  # doing 2d lookup as [[index1.row, index2.row],[index1.column, index2.column]]
        resultH = self.reverse_lookup[([rowIndexer], [valuesH])]  # where 2d index tuple would be (index1.row, index1.column)
    # lookup int or do linear interpolation
        return resultL * (isIntMask + values - valuesL) + resultH * (valuesH - values)    

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data * (1 + self.epsilon)
        res = jax.vmap(self._reverse_emp_integral_trans_single)(data)
        # res = [self._reverse_emp_integral_trans_single(row) for row in data]
        return res[:, 0, :]

def load_data(data_set, n_train=None, n_test=None):
    train = np.load(data_set + '.npy')
    if n_train is not None:
        train = train[:n_train]
    return train, []


