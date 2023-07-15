import autograd.numpy as anp
import numpy as np
import scipy.stats


class ObservationModel:
    def __init__(self, use_gp: bool = False) -> None:
        self.use_gp = use_gp
        if not self.use_gp:
            print("WARNING: observation model will NOT use GP to compute likelihood.")

    def compute_likelihood(self, dz: np.ndarray, Q: np.ndarray) -> float:
        """Compute the likelihood of an observation given its innovation properties.

        This method uses either pre-configured observation covariances or computes them
        using a gaussian process conditioned on the innovation distances.
        """
        # TODO: implement GP structure
        # this likely involves acceping multiple ianputs at once
        # and returning multiple outputs to speed up inference
        # moreover, it also requires the GP regression method to return variances as well...
        # NOTE: should we not use (H Q H.T) + GP_cov instead of only GP_cov?!
        return ObservationModel.likelihood_from_innovation(dz, Q)

    @staticmethod
    def range_bearing_observation_landmark(l: anp.ndarray, x: anp.ndarray) -> anp.ndarray:
        lx, ly, lz = l
        px, py, pyaw = x
        # position delta components
        dx = lx - px
        dy = ly - py
        dz = lz
        rho = anp.linalg.norm(anp.array([dx, dy, dz]))
        phi = anp.mod(anp.arctan2(dy, dx) - pyaw, 2 * anp.pi)
        psi = anp.arcsin(dz / rho)
        return anp.array([rho, phi, psi], dtype=anp.float64)

    @staticmethod
    def range_bearing_observation_keypoint(p: anp.ndarray) -> anp.ndarray:
        rho = anp.linalg.norm(p)
        phi = anp.arctan2(p[1], p[0])
        psi = anp.arcsin(p[2] / rho)
        return anp.array([rho, phi, psi], dtype=anp.float64)

    @staticmethod
    def likelihood_from_innovation(dz: np.ndarray, Q: np.ndarray) -> float:
        return scipy.stats.multivariate_normal.pdf(x=dz, mean=np.zeros(3), cov=Q)


if __name__ == "__main__":
    # test automatic differentiation:
    from autograd import jacobian

    # simulate robot pose
    x = anp.array([0.5, 0.2, anp.deg2rad(30)])
    # simulate ladnmark position
    p_l = anp.array([3, 2, 1], dtype=anp.float64)

    def h(l):
        return ObservationModel.range_bearing_observation_landmark(l, x)

    jacobian_of_h = jacobian(h)
    # compute jacobian of h at l
    Jh = jacobian_of_h(p_l)
    print(Jh)
