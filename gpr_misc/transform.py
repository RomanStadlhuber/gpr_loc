from rosbags.typesys.types import nav_msgs__msg__Odometry as Odometry
from scipy.spatial.transform import Rotation
from typing import Optional
import numpy as np


class Pose2D:
    def __init__(self, T0: Optional[np.ndarray] = None) -> None:
        self.T = T0 if T0 is not None else np.eye(3)

    def as_twist(self) -> np.ndarray:
        """Convert this pose to a global twist"""
        return Pose2D.pose_to_twist(self.T)

    def perturb(self, u: np.ndarray) -> None:
        dT = Pose2D.twist_to_pose(u)
        self.T = self.T @ dT

    def inv(self) -> np.ndarray:
        return Pose2D.invert_pose(self.T)

    def as_t3d(self) -> np.ndarray:
        """Obtain the transform as as 4x4 matrix in 3D space."""
        R = self.T[:2, :2]
        t = self.T[:2, 2].reshape(-1, 1)
        T = np.block(
            [
                [R, np.zeros((2, 1)), t],
                [np.array([0, 0, 1, 0])],
                [np.array([0, 0, 0, 1])],
            ]
        )
        return T

    @staticmethod
    def from_twist(x: np.ndarray) -> "Pose2D":
        T = Pose2D.twist_to_pose(x)
        return Pose2D(T)

    @staticmethod
    def twist_to_pose(twist: np.ndarray) -> np.ndarray:
        """Converts a 3-dimensional twist vector to a 2D affine transformation."""
        x = twist[0]
        y = twist[1]
        w = twist[2]
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
                [0, 0, 1],
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

    @staticmethod
    def from_odometry(odom: Odometry) -> "Pose2D":
        """Compute a 2D pose from a `nav_msgs/Odometry` message."""
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        rotation = Rotation.from_quat(
            [
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ]
        )
        theta, *_ = rotation.as_euler("zyx", degrees=False)

        u = np.array([x, y, theta], dtype=np.float64)
        return Pose2D.from_twist(u)
