import autograd.numpy as np


class ObservationModel:
    @staticmethod
    def range_bearing_observation_landmark(l: np.ndarray, x: np.ndarray) -> np.ndarray:
        lx, ly, lz = l
        px, py, pyaw = x
        # position delta components
        dx = lx - px
        dy = ly - py
        dz = lz
        rho = np.linalg.norm(np.array([dx, dy, dz]))
        phi = np.mod(np.arctan2(dy, dx) - pyaw, 2 * np.pi)
        psi = np.arcsin(dz / rho)
        return np.array([rho, phi, psi], dtype=np.float64)

    @staticmethod
    def range_bearing_observation_keypoint(p: np.ndarray) -> np.ndarray:
        rho = np.linalg.norm(p)
        phi = np.arctan2(p[1], p[0])
        psi = np.arcsin(p[2] / rho)
        return np.array([rho, phi, psi], dtype=np.float64)


if __name__ == "__main__":
    # test automatic differentiation:
    from autograd import jacobian

    # simulate robot pose
    x = np.array([0.5, 0.2, np.deg2rad(30)])
    # simulate ladnmark position
    p_l = np.array([3, 2, 1], dtype=np.float64)

    def h(l):
        return ObservationModel.range_bearing_observation_landmark(l, x)

    jacobian_of_h = jacobian(h)
    # compute jacobian of h at l
    Jh = jacobian_of_h(p_l)
    print(Jh)
