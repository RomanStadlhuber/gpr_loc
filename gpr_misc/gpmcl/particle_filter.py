from rosbags.typesys.types import nav_msgs__msg__Odometry as Odometry
from scipy.spatial.transform import Rotation
from gpmcl.mapper import FeatureMap3D
from dataclasses import dataclass
from typing import Optional
import numpy as np


class Pose2D:
    def __init__(self, T0: Optional[np.ndarray] = None) -> None:
        self.T = T0 or np.eye(3)

    def as_twist(self) -> np.ndarray:
        """Convert this pose to a global twist"""
        return Pose2D.pose_to_twist(self.T)

    def perturb(self, u: np.ndarray) -> None:
        dT = Pose2D.twist_to_pose(u)
        self.T = dT @ self.T

    def inv(self) -> np.ndarray:
        return Pose2D.invert_pose(self.T)

    @staticmethod
    def from_twist(x: np.ndarray) -> "Pose2D":
        T = Pose2D.twist_to_pose(x)
        return Pose2D(T)

    @staticmethod
    def twist_to_pose(twist: np.ndarray) -> np.ndarray:
        """Converts a 3-dimensional twist vector to a 2D affine transformation."""
        x, y, w = twist
        T = np.array(
            [
                [np.cos(w), -np.sin(w), x],
                [np.sin(w), np.cos(w), y],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        return T

    @staticmethod
    def invert_pose(T: np.ndarray) -> np.ndarray:
        """Inverts a 2D affine transformation."""
        R = T[:2, :2]
        t = np.reshape(T[:2, 2], (-1, 1))
        T_inv = np.block(
            [
                [R.T, -R.T @ t],
                [
                    [0, 0, 1],
                ],
            ]
        )
        return T_inv

    @staticmethod
    def pose_to_twist(T: np.ndarray) -> np.ndarray:
        """Converts a 2D affine transformation into a twist vector of shape `(x, y, theta)`"""
        R = T[:2, :2]
        sin = R[1, 0]
        cos = R[0, 0]
        x = T[0, 2]
        y = T[1, 2]
        theta = np.arctan2(sin, cos)
        return np.array([x, y, theta], dtype=np.float64)


@dataclass
class FeaturesAndMap:
    observed_features: FeatureMap3D
    map: FeatureMap3D


class ParticleFilter:
    def __init__(self, T0: Pose2D, N: int, R: np.ndarray, Q: np.ndarray) -> None:
        self.N = N
        self.R = R
        self.Q = Q
        # sample the initial states
        # an N x 3 array representing the particles as twists
        self.Xs = self.__sample_multivariate_normal(T0)
        # the last pose deltas as twists, required for GP inference
        self.dX_last = np.zeros((N, 3), dtype=np.float64)

    def predict(self, U: Odometry) -> None:
        x = U.pose.pose.position.x
        y = U.pose.pose.position.y
        rotation = Rotation.from_quat(
            [
                U.pose.pose.orientation.x,
                U.pose.pose.orientation.y,
                U.pose.pose.orientation.z,
                U.pose.pose.orientation.w,
            ]
        )
        theta, *_ = rotation.as_euler("zyx", degrees=False)

        u = np.array([x, y, theta], dtype=np.float64)
        X_est = Pose2D.from_twist(u)

        def get_estimated_delta_x(x: np.ndarray) -> np.ndarray:
            # inverse affine transform of current pose
            T_x_inv = Pose2D.from_twist(x).inv()
            # affine transform of estimated pose
            T_est = X_est.T
            # estimated delta motion
            T_delta = T_x_inv @ T_est
            # compute the 2D rotation frmm the y and x components of the R mat using atan2d
            sin_yaw = T_delta[1, 0]
            cos_yaw = T_delta[0, 0]
            yaw_delta = np.arctan2(sin_yaw, cos_yaw)  # x = atan2d(y=sin(x), x=cos(x))
            x_delta = T_delta[0, 2]
            y_delta = T_delta[1, 2]
            return np.array([x_delta, y_delta, yaw_delta])

        # N x 3 array of estimated motion
        dX_est = np.array(list(map(get_estimated_delta_x, self.Xs)))
        # GP inference features
        X = np.hstack((self.dX_last, dX_est))
        # TODO: run GP inference in bulk and apply to each particle

        pass

    def update(self, Z: FeaturesAndMap) -> None:
        # TODO: how to incorporate landmark measurements?!

        # TODO: don't forget to update self.dX_last!!
        pass

    def __sample_multivariate_normal(self, X0: Pose2D) -> np.ndarray:
        """Sample particles from a multivariate normal.

        Uses the process noise covariance `R`.
        Returns an `N x DIM` array representing the sampled particles.
        """
        x0 = X0.as_twist()
        # the particle states
        Xs = np.random.default_rng().multivariate_normal(x0, self.R, (self.N, 3))
        return Xs
