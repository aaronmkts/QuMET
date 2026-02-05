"""Utility functions for manual dataset transformations.

This module provides utilities for discretization and normalization of
manual quantum dataset samples.
"""

from .discretisation import compute_discretization
from .normalisers import MinMaxNormalizer, PITNormalizer
