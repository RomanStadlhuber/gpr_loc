import numpy as np


def range_bearing_observation_landmark(l: np.ndarray, x: np.ndarray) -> np.ndarray:
    lx, ly, lz = l
    px, py, pyaw = x
    # position delta components
    dx = lx - px
    dy = ly - py
    dz = lz
    rho = np.linalg.norm([dx, dy, dz])
    phi = np.mod(np.arctan2(dy, dx) - pyaw, 2 * np.pi)
    psi = np.arcsin(dz / rho)
    return np.array([rho, phi, psi], dtype=np.float64)


def range_bearing_observation_keypoint(p: np.ndarray) -> np.ndarray:
    rho = np.linalg.norm(p)
    phi = np.arctan2(p[1], p[0])
    psi = np.arcsin(p[2] / rho)
    return np.array([rho, phi, psi], dtype=np.float64)
