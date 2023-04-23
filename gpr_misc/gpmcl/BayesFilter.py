from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import scipy.linalg


class State(ABC):
    """Base class for estimated states."""

    @abstractmethod
    def __init__(self, dim: int, x0: Optional[np.ndarray]) -> None:
        """Initialize the state."""
        if x0 is not None:
            d, *_ = np.shape(x0)
            if d != dim:
                raise (Exception(f"Dimensionality {d} of initial state {x0} does not match dim: {dim}."))

        self.x = x0 or np.zeros((dim), dtype=np.float64)

    @abstractmethod
    def perturb(self, u: np.ndarray) -> None:
        """Perturb the state by a control vector `u`."""
        pass


class Observation(ABC):
    def __init__(self) -> None:
        pass


class BayesFilter(ABC):
    """Base class for Bayesian filtering."""

    @abstractmethod
    def __init__(self, x0: State) -> None:
        pass

    @abstractmethod
    def predict(self, u: np.ndarray) -> None:
        """Predict the next state based on some input `u`."""
        pass

    @abstractmethod
    def update(self, z: Observation) -> None:
        """Update the state based on an observation `z`."""
        pass
