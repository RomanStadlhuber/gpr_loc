from dataclasses import dataclass
from typing import Optional, Tuple
from gpmcl.observation_model import ObservationModel
from gpmcl.transform import Pose2D
from filterpy.kalman import ExtendedKalmanFilter
from autograd import jacobian
import numpy as np
import numpy.typing as npt
import scipy.stats
import open3d


@dataclass
class FastSLAMParticle:
    """A particle as described in the Fast-SLAM algorithm."""

    # 2d pose
    x: Pose2D
    # T x 3 list of all poses
    trajectory: np.ndarray = np.empty((0, 3))
    # whether the landmarks of this particle have already been initialized
    landmarks_initialized: bool = False
    # N x 3 positions of all landmarks in the map
    landmarks: np.ndarray = np.empty((0, 3))
    # N x (3 x 3) landmark position covariance
    landmark_covariances: np.ndarray = np.empty((0, 3, 3))
    # counts how often a landmark is observed during its lifetime
    observation_counter: np.ndarray = np.zeros((0, 1), dtype=np.int32)

    @staticmethod
    def copy(other: "FastSLAMParticle") -> "FastSLAMParticle":
        return FastSLAMParticle(
            x=Pose2D(np.copy(other.x.T)),
            trajectory=np.copy(other.trajectory),
            landmarks_initialized=other.landmarks_initialized,
            landmarks=np.copy(other.landmarks),
            landmark_covariances=np.copy(other.landmark_covariances),
            observation_counter=np.copy(other.observation_counter),
        )

    def has_map(self) -> bool:
        """Evaluate whether or not this particle already has a map."""
        return self.landmarks_initialized

    def apply_u(self, u: np.ndarray, R: np.ndarray = np.eye(3, dtype=np.float64)) -> None:
        """Apply a motion to the particles pose.

        NOTE: this implicitly updates the trajectory of the particle.
        """
        # add the current pose to the trajectory
        x_vec = self.x.as_twist()
        self.trajectory = np.vstack((self.trajectory, x_vec))
        # generate motion noise from covariance
        noise = scipy.stats.multivariate_normal.rvs(mean=np.zeros(3), cov=R)
        # apply the motion to obtain the new pose
        self.x.perturb(u + noise)

    def estimate_correspondences(
        self,
        pcd_keypoints: open3d.geometry.PointCloud,
        knn_search_radius: float = 0.6,
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Estimate the correspondences between observed keypoints and landmarks in the map.

        returns `(all_correspondences, pcd_keypoints_in_map_frame, best_correspondence, new_landmark_idxs)`.

        ### Remarks

        This creates a copy of `pcd_keypoints` in order to transform the keypoints into the map frame and
        perofrm KDTree radius search on them.
        """
        # copy the original pointcloud
        pcd_keypoints = open3d.geometry.PointCloud(pcd_keypoints)
        num_keypoints = np.asarray(pcd_keypoints.points).shape[0]
        idxs_all_keypoints = np.linspace(start=0, stop=num_keypoints - 1, num=num_keypoints, dtype=np.int32)
        # transform the keypoints into the map frame
        # this is done under the assumptions that, at some point, there are more landmarks than keypoints being observed
        pcd_keypoints.transform(self.x.as_t3d())
        n_landmarks, *_ = self.landmarks.shape
        if n_landmarks == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),  # closest correspondence
                idxs_all_keypoints,  # previously unseen correspondences
            )
        else:
            # create a KD-Tree for correspondence search
            kdtree_keypoints = open3d.geometry.KDTreeFlann(pcd_keypoints)
            correspondences = np.empty((0, 2), dtype=np.int32)
            c_distances = np.empty((0, 1), dtype=np.float64)
            for idx_l, landmark in enumerate(self.landmarks):
                # (num_neighbors, idxs, distances) = ...
                # see: http://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html#open3d.geometry.KDTreeFlann.search_radius_vector_3d
                [num_matches, idxs, distances] = kdtree_keypoints.search_radius_vector_3d(
                    query=landmark, radius=knn_search_radius
                )
                # store distances and correspondences only when they're available
                if num_matches > 0:
                    idxs = np.asarray(idxs, dtype=np.int32)
                    distances = np.asarray(distances, dtype=np.float64)
                    idx_min_dist = np.argmin(distances)
                    # set min-distance point to be the corresponding
                    correspondences = np.vstack((correspondences, [idx_l, idxs[idx_min_dist]]))
                    c_distances = np.vstack((c_distances, distances[idx_min_dist]))
            # safeguard in case there are no correspondences
            if c_distances.shape[0] == 0:
                return (
                    np.empty((0, 2), dtype=np.int32),
                    np.empty((0, 2), dtype=np.int32),  # closest correspondence
                    idxs_all_keypoints,  # previously unseen correspondences
                )
            else:
                # get the correspondence with the lowest distance
                # this is considered the most likely correspondence
                idx_c_min_distance = np.argmin(c_distances)
                closest_corresondence = correspondences[idx_c_min_distance]
                # previously unseen keypoints are those whose indices are not in the correspondence list
                matched_keypoints = set(correspondences[:, 1])
                unmatched_keypoints = np.array(
                    list(set(idxs_all_keypoints).difference(matched_keypoints)),
                    dtype=np.int32,
                )
                idxs_matched_landmarks = correspondences[:, 0]
                num_landmarks = self.landmarks.shape[0]
                idxs_all_landmarks = np.linspace(start=0, stop=num_landmarks - 1, num=num_landmarks, dtype=np.int32)
                # TODO: can we assume unique? (see param "assume_unique"), in
                # https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html
                idxs_unmatched_landmarks = np.setdiff1d(idxs_all_landmarks, idxs_matched_landmarks)
                # increment the observation counter for matched landmarks
                self.observation_counter[idxs_matched_landmarks, 0] += 1
                # decrement the observation counter for unmatched landmarks
                self.observation_counter[idxs_unmatched_landmarks, 0] -= 1
                return (correspondences, closest_corresondence, unmatched_keypoints)

    def add_new_landmarks_from_keypoints(
        self,
        idxs_new_landmarks: npt.NDArray[np.int32],
        keypoints_in_robot_frame: npt.NDArray[np.float64],
        position_covariance: npt.NDArray[np.float64],
        max_active_landmarks: int = 5,
    ) -> None:
        # region: add all keypoints to map if there is no restriction
        if max_active_landmarks < 0:
            pcd_l_new = open3d.geometry.PointCloud(
                open3d.utility.Vector3dVector(keypoints_in_robot_frame[idxs_new_landmarks])
            )
            # transform the keyponits into the map frame
            pcd_l_new.transform(self.x.as_t3d())
            l_new = np.asarray(pcd_l_new.points)
            self.__add_landmarks(ls=l_new, Q_0=position_covariance)
            self.landmarks_initialized = True
        # endregion
        # region: add only the remainder of allowed landmarks
        # amount of new landmarks that can be admitted
        num_new = max_active_landmarks - self.landmarks.shape[0]
        if num_new > 0:
            pcd_l_new = open3d.geometry.PointCloud(
                open3d.utility.Vector3dVector(keypoints_in_robot_frame[idxs_new_landmarks][:num_new])
            )
            # transform the keyponits into the map frame
            pcd_l_new.transform(self.x.as_t3d())
            l_new = np.asarray(pcd_l_new.points)
            self.__add_landmarks(ls=l_new, Q_0=position_covariance)
            self.landmarks_initialized = True
        # endregion

    def update_existing_landmarks(
        self,
        correspondences: npt.NDArray[np.int32],
        keypoints_in_robot_frame: npt.NDArray[np.float64],
        observation_covariance: np.ndarray,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Update the landmarks from observed keypoints and their landmark correspondences.

        Returns the `(observation_errors, innovation_covariances)` of the respective innovations.

        ### Remark
        - Retured values are ordered w.r.t. the particles landmarks.
        - Observation error `d_z` and innovation covariance `Q` might be used to compute a particles likelihood.
        """
        Q_z = observation_covariance
        # errors and covariances of the landmark observations
        # these are used to compute particle likelihoods
        N_landmarks, *_ = self.landmarks.shape
        ds = np.empty((N_landmarks, 3), dtype=np.float64)
        Qs = np.empty((N_landmarks, 3, 3), dtype=np.float64)
        # update the landmarks using EKF approach
        for idx_l, idx_kp in correspondences:
            # the observed keypoint in XYZ coordinates
            kp = keypoints_in_robot_frame[idx_kp]
            # landmark range-bearing observation and its corresponding jacobian
            z_l, H_l = self.__compute_landmark_observation(idx_l)
            # keypoint range-bearing observation
            z_kp = ObservationModel.range_bearing_observation_keypoint(kp)
            # compute delta between the two observation
            delta_z = ObservationModel.observation_delta(z_true=z_kp, z_est=z_l)
            # covariance of the landmark in question
            S_l = self.landmark_covariances[idx_l]
            # innovation covariance
            # uses cholesky factor because H @ S @ H.T would not be symmetric
            # ... why?
            C = np.linalg.cholesky(S_l)
            HC = H_l @ C
            Q_l = HC @ HC.T + Q_z
            # kalman gain
            K_l = S_l @ H_l.T @ np.linalg.inv(Q_l)
            # update the landmark in question
            self.__update_landmark(idx_l, delta=delta_z, K_gain=K_l, H=H_l, Qz=Q_z)
            # set observation error and its covariance
            ds[idx_l] = delta_z
            Qs[idx_l] = Q_l
        # return the errors and covariances for likelihood computation
        return (ds, Qs)

    def prune_landmarks(self, max_distance: float, max_unobserved_count: int = -1) -> None:
        """Removes landmarks that exceed the mapping range or are unobserved too often.

        This is basically a post-update map-management method.
        """
        robot_position = np.array([*self.x.as_twist()[:2], 0], dtype=np.float64)
        relative_landmark_positions = np.copy(self.landmarks)
        relative_landmark_positions -= robot_position
        landmark_distances = np.linalg.norm(relative_landmark_positions, axis=1)
        idxs_landmarks_in_range = np.where(landmark_distances <= max_distance)
        idxs_landmarks_unobserved_too_often = np.where(self.observation_counter <= max_unobserved_count)
        idxs_to_keep = np.setdiff1d(idxs_landmarks_in_range, idxs_landmarks_unobserved_too_often)
        # keep all the map-related values of the landmarks in range and aren't unobserved too often
        self.landmarks = self.landmarks[idxs_to_keep]
        self.landmark_covariances = self.landmark_covariances[idxs_to_keep]
        self.observation_counter = self.observation_counter[idxs_to_keep]

    def __add_landmarks(self, ls: np.ndarray, Q_0: np.ndarray, Qs: Optional[np.ndarray] = None) -> None:
        """Add a set of landmarks with common or individual covariances.

        Pass `Qs` as individual landmark covariances.
        """
        num_new_landmarks = ls.shape[0]
        self.landmarks = np.vstack((self.landmarks, ls))
        self.observation_counter = np.vstack(
            (self.observation_counter, np.ones((num_new_landmarks, 1), dtype=np.int32))
        )
        if Qs is None:
            self.landmark_covariances = np.vstack(
                (self.landmark_covariances, np.repeat([Q_0], repeats=len(ls), axis=0))
            )
        else:
            self.landmark_covariances = np.vstack((self.landmark_covariances, Qs))

    def __update_landmark(self, idx: int, delta: np.ndarray, K_gain: np.ndarray, H: np.ndarray, Qz: np.ndarray) -> None:
        """Update the position of an individual landmark given its index and Kalman filter values"""
        self.landmarks[idx] += K_gain @ delta
        # update landmark position mean
        J = np.eye(3) - K_gain @ H
        P = self.landmark_covariances[idx]
        # update landmark position covariance
        self.landmark_covariances[idx] = J @ P @ J.T + K_gain @ Qz @ K_gain.T

    def __compute_landmark_observation(self, idx_landmark: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the observation of a landmark plus the corresponding jacobian w.r.t. the landmark.

        Returns `(z, H)` where `z` is the observation vector and `H` the models jacobian w.r.t. the landmark `l`.
        """

        def h(l: np.ndarray) -> np.ndarray:
            return ObservationModel.range_bearing_observation_landmark(l=l, x=self.x.as_twist())

        l_i = self.landmarks[idx_landmark]
        z_i = h(l_i)
        jacobian_of_h = jacobian(h)
        H_i = jacobian_of_h(l_i)

        return (z_i, H_i)
