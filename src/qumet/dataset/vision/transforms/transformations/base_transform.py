from abc import ABC, abstractmethod


class TransformBase(ABC):
    @abstractmethod
    def fit(self, data):
        """Fit the transformation model on the data."""
        pass

    @abstractmethod
    def __call__(self, x):
        """Apply the forward transformation."""
        pass

    @abstractmethod
    def inverse_transform(self, x):
        """Apply the inverse transformation."""
        pass
