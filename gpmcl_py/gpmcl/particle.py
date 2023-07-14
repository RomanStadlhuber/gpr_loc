from dataclasses import dataclass
from typing import Optional, Tuple
from gpmcl.observation_model import range_bearing_observation_landmark, range_bearing_observation_keypoint
from gpmcl.transform import Pose2D
from autograd import jacobian
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

    def estimate_correspondences(
        self, pcd_keypoints: open3d.geometry.PointCloud, max_distance: float = 0.6
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Estimate the correspondences between observed keypoints and landmarks in the map.

        returns `(all_correspondences, pcd_keypoints_in_map_frame, best_correspondence, new_landmark_idxs)`.

        ### Remarks

        This creates a copy of `pcd_keypoints` in order to transform the keypoints into the map frame and
        perofrm KDTree radius search on them.
        """
        # copy the original pointcloud
        pcd_keypoints = open3d.geometry.PointCloud(pcd_keypoints)
        idxs_all_keypoints = np.linspace(0, np.asarray(pcd_keypoints).shape[0], dtype=np.int32)
        # transform the keypoints into the map frame
        # this is done under the assumptions that, at some point, there are more landmarks than keypoints being observed
        pcd_keypoints.transform()
        n_landmarks, *_ = self.landmarks.shape
        if n_landmarks == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),  # closest correspondence
                idxs_all_keypoints,  # previously unseen correspondences
            )
        else:
            # create a KD-Tree for correspondence search
            kdtree_keypoints = open3d.geometry.KDTreeFLANN(pcd_keypoints)
            correspondences = np.empty((0, 2), dtype=np.int32)
            c_distances = np.empty(n_landmarks, dtype=np.float64)
            for idx_l, landmark in enumerate(self.landmarks):
                # (num_neighbors, idxs, distances) = ...
                # see: http://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html#open3d.geometry.KDTreeFlann.search_radius_vector_3d
                [_, idxs, distances] = kdtree_keypoints.search_radius_vector3d(query=landmark, radius=max_distance)
                idx_min_dist = np.argmin(distances)
                # set min-distance point to be the corresponding
                correspondences = np.vstack((correspondences, [idx_l, idxs[idx_min_dist]]))
                c_distances[idx_l] = distances[idx_min_dist]
            # get the correspondence with the lowest distance
            # this is considered the most likely correspondence
            idx_c_min_distance = np.argmin(c_distances)
            closest_corresondence = correspondences[idx_c_min_distance]
            # previously unseen keypoints are those whose indices are not in the correspondence list
            ixds_previously_unseen_kps = np.unique(np.vstack((idxs_all_keypoints, correspondences[:, 1])))
            # TODO: obtain idxs of previously unseen features!!
            return (correspondences, closest_corresondence, ixds_previously_unseen_kps)

    def add_new_landmarks_from_keypoints(
        self,
        idxs_new_landmarks: npt.NDArray[np.int32],
        keypoints_in_robot_frame: npt.NDArray[np.float64],
        position_covariance: npt.NDArray[np.float64],
    ) -> None:
        pcd_l_new = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(keypoints_in_robot_frame[idxs_new_landmarks])
        )
        # transform the keyponits into the map frame
        pcd_l_new.transform(self.x.as_t3d())
        l_new = np.asarray(pcd_l_new.points)
        self.__add_landmarks(ls=l_new, Q_0=position_covariance)

    def update_existing_landmarks(
        self,
        correspondences: npt.NDArray[np.int32],
        keypoints_in_robot_frame: npt.NDArray[np.float64],
        observation_covariance: np.ndarray,
    ) -> None:
        Q_z = observation_covariance
        for idx_l, idx_kp in correspondences:
            # observation and its corresponding jacobian
            kp = keypoints_in_robot_frame[idx_kp]
            z_l, H_l = self.__compute_landmark_observation(idx_l)
            z_kp = range_bearing_observation_keypoint(kp)
            delta_z = z_l - z_kp
            Q_l = self.landmark_covariances[idx_l]
            K_l = Q_l @ H_l.T @ np.linalg.inv(H_l @ Q_l @ H_l.T + Q_z)
            self.__update_landmark(idx_l, delta=delta_z, K_gain=K_l, H=H_l, Qz=Q_z)

    def register_unobserved_landmarks(self, idxs_unobserved: npt.NDArray[np.int32]) -> None:
        # TODO: implement
        pass

    def __add_landmarks(self, ls: np.ndarray, Q_0: np.ndarray, Qs: Optional[np.ndarray] = None) -> None:
        """Add a set of landmarks with common or individual covariances.

        Pass `Qs` as individual landmark covariances.
        """
        self.landmarks = np.vstack((self.landmarks, ls))
        if Qs is None:
            self.landmark_covariances = np.vstack((self.landmark_covariances, np.repeat([Q_0], repeats=len(ls))))
        else:
            self.landmark_covariances = np.vstack((self.landmark_covariances, Qs))

    def __update_landmark(self, idx: int, delta: np.ndarray, K_gain: np.ndarray, H: np.ndarray, Qz: np.ndarray) -> None:
        """Update the position of an individual landmark given its index and Kalman filter values"""
        # TODO: normalize the angle values of the delta vector to [-pi, pi]!
        self.landmarks[idx] += K_gain @ delta
        # update landmark position mean
        J = np.eye(3) - K_gain * H
        P = self.landmark_covariances[idx]
        # update landmark position covariance
        self.landmark_covariances[idx] = J @ P @ J.T + K_gain @ Qz @ K_gain.T

    def __compute_landmark_observation(self, idx_landmark: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the observation of a landmark plus the corresponding jacobian w.r.t. the landmark.

        Returns `(z, H)` where `z` is the observation vector and `H` the models jacobian w.r.t. the landmark `l`.
        """

        def h(l: np.ndarray) -> np.ndarray:
            return range_bearing_observation_landmark(l=l, x=self.x.as_twist())

        l_i = self.landmarks[idx_landmark]
        z_i = h(l_i)
        jacobian_of_h = jacobian(h)
        H_i = jacobian_of_h(l_i)

        return (z_i, H_i)
