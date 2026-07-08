"""Tests for manual bitstring dataset helpers."""

import importlib

import numpy as np


def test_bars_and_stripes_module_is_importable_and_generates_unique_patterns():
    module = importlib.import_module("qumet.dataset.manual.bitstrings.bars_and_stripes")

    data = module.bars_and_stripes(2, 3)

    assert isinstance(data, np.ndarray)
    assert data.shape == (10, 6)
    assert np.unique(data, axis=0).shape == data.shape
    assert np.array_equal(data[0], np.zeros(6, dtype=int))
    assert np.array_equal(data[-1], np.ones(6, dtype=int))
