from dataclasses import dataclass
from typing import List, Optional, Tuple
from gpmcl.transform import Pose2D
import numpy as np
import numpy.typing as npt
import open3d


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

    def update_landmark(self, idx: int, delta: np.ndarray, K_gain: np.ndarray, H: np.ndarray, Qz: np.ndarray) -> None:
        """Update the position of an individual landmark given its index and Kalman filter values"""
        # TODO: normalize the angle values of the delta vector to [-pi, pi]!
        self.landmarks[idx] += K_gain @ delta
        # update landmark position mean
        J = np.eye(3) - K_gain * H
        P = self.landmark_covariances[idx]
        # update landmark position covariance
        self.landmark_covariances[idx] = J @ P @ J.T + K_gain @ Qz @ K_gain.T

    def add_landmarks(self, ls: List[np.ndarray], Q_0: np.ndarray, Qs: Optional[np.ndarray] = None) -> None:
        """Add a set of landmarks with common or individual covariances.

        Pass `Qs` as individual landmark covariances.
        """
        self.landmarks = np.vstack((self.landmarks, ls))
        if Qs is None:
            self.landmark_covariances = np.vstack((self.landmark_covariances, np.repeat([Q_0], repeats=len(ls))))
        else:
            self.landmark_covariances = np.vstack((self.landmark_covariances, Qs))

    def estimate_correspondences(
        self, pcd_keypoints: open3d.geometry.PointCloud, max_distance: float = 0.6
    ) -> Tuple[npt.NDArray[np.int32], open3d.geometry.PointCloud]:
        """Estimate the correspondences between observed keypoints and landmarks in the map.

        ### Remarks

        This creates a copy of `pcd_keypoints` in order to transform the keypoints into the map frame and
        perofrm KDTree radius search on them.
        """
        # copy the original pointcloud
        pcd_keypoints = open3d.geometry.PointCloud(pcd_keypoints)
        # transform the keypoints into the map frame
        # this is done under the assumptions that, at some point, there are more landmarks than keypoints being observed
        pcd_keypoints.transform()
        n_landmarks, *_ = self.landmarks.shape
        if n_landmarks == 0:
            return (np.empty((0, 2), dtype=np.int32), pcd_keypoints)
        else:
            # create a KD-Tree for correspondence search
            kdtree_keypoints = open3d.geometry.KDTreeFLANN(pcd_keypoints)
            correspondences = np.empty((n_landmarks, 2), dtype=np.int32)
            for idx_l, landmark in enumerate(self.landmarks):
                # (num_neighbors, idxs, distances) = ...
                # see: http://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html#open3d.geometry.KDTreeFlann.search_radius_vector_3d
                [_, idxs, distances] = kdtree_keypoints.search_radius_vector3d(query=landmark, radius=max_distance)
                idx_min_dist = np.argmin(distances)
                # set min-distance point to be the corresponding
                correspondences[idx_l] = [idx_l, idxs[idx_min_dist]]
            return (correspondences, pcd_keypoints)
