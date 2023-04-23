from gpmcl.BayesFilter import State, Observation, BayesFilter
from gpmcl.mapper import FeatureMap3D
from typing import Optional
import numpy as np
import scipy.linalg


class PoseSE2(State):
    def __init__(self, dim: int = 3, x0: Optional[np.ndarray] = None) -> None:
        super().__init__(dim, x0)

        self.x = scipy.linalg.expm(x0 or np.zeros(3))

    def perturb(self, u: np.ndarray) -> None:
        m = PoseSE2.__vee(u)  # tangent vector to tangent matrix
        self.x = self.x @ scipy.linalg.expm(m)

    @staticmethod
    def __vee(u: np.ndarray) -> np.ndarray:
        t = np.reshape(u[:2], (-1, 1))  # x and y tangent movement
        w = u[2]  # tangent rotation
        return np.block([[np.diag([-w, w]), t], [np.zeros((1, 3), dtype=np.float64)]])


class FeaturesAndMap(Observation):
    def __init__(self, observed_features: FeatureMap3D, map: FeatureMap3D) -> None:
        self.observed_features = observed_features
        self.map = map


class ParticleFilter(BayesFilter):
    def __init__(self, x0: State, N: int) -> None:
        self.N = N
        # TODO initialize particles from x0!

    def predict(self, u: np.ndarray) -> None:
        # TODO: run GP inference using x' = u + w (white noise)
        # update each particle by x.perturb(x')

        pass

    def update(self, z: FeaturesAndMap) -> None:
        # TODO: how to incorporate landmark measurements?!
        pass
