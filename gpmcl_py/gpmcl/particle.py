from dataclasses import dataclass
from typing import List, Optional
from gpmcl.transform import Pose2D
import numpy as np


@dataclass
class FastSLAMParticle:
    """A particle as described in the Fast-SLAM algorithm."""

    # 2d pose
    x: Pose2D
    # T x 3 list of all poses
    trajectory: np.ndarray = np.empty((0, 3))
    # N x 3 positions of all landmarks in the map
    landmarks: np.ndarray = np.empty((0, 3))
    # N x (3 x 3) landmark position covariance
    landmark_covariances: np.ndarray = np.empty((0, 3, 3))

    def apply_u(self, u: np.ndarray) -> None:
        """Apply a motion to the particles pose.

        NOTE: this implicitly updates the trajectory of the particle.
        """
        # add the current pose to the trajectory
        x_vec = self.x.as_twist()
        self.trajectory = np.vstack((self.trajectory, x_vec))
        # apply the motion to obtain the new pose
        self.x.perturb(u)

    def update_landmark(self, idx: int, delta: np.ndarray, K_gain: np.ndarray, H: np.ndarray, Q: np.ndarray) -> None:
        """Update the position of an individual landmark given its index and Kalman filter values"""
        # TODO: normalize the angle values of the delta vector to [-pi, pi]!
        self.landmarks[idx] += K_gain @ delta
        # update landmark position mean
        J = np.eye(3) - K_gain * H
        P = self.landmark_covariances[idx]
        # update landmark position covariance
        self.landmark_covariances[idx] = J @ P @ J.T + K_gain @ Q @ K_gain.T

    def add_landmarks(self, ls: List[np.ndarray], Q_0: np.ndarray, Qs: Optional[np.ndarray] = None) -> None:
        """Add a set of landmarks with common or individual covariances.

        Pass `Qs` as individual landmark covariances.
        """
        self.landmarks = np.vstack((self.landmarks, ls))
        if Qs is None:
            self.landmark_covariances = np.vstack((self.landmark_covariances, np.repeat([Q_0], repeats=len(ls))))
        else:
            self.landmark_covariances = np.vstack((self.landmark_covariances, Qs))
