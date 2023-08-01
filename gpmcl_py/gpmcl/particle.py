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
        max_observation_range: float = 10.0,
    ) -> Tuple[
        npt.NDArray[np.int32],  # [[idx_l, idx_f], ...] update indices
        npt.NDArray[np.int32],  # [idx_l, idx_f] closest correspondence
        npt.NDArray[np.int32],  # [[idx_l, idx_f], ...]  previously unobserved keypoints
    ]:
        """Estimate the correspondences between observed keypoints and landmarks in the map.

        returns `(all_correspondences, pcd_keypoints_in_map_frame, best_correspondence, new_landmark_idxs)`.

        ### Remarks

        This creates a copy of `pcd_keypoints` in order to transform the keypoints into the map frame and
        perofrm KDTree radius search on them.

        Moreover, this method internally updates the observation-counter of all landmarks.
        Counters of landmarks that matched to a feature are incremented.
        Counters of landmarks that are within the `max_observation_range` but are unmatched are decremented.

        Use the `prune_landmarks()` method to remove landmarks that are unobserved too often or outside the mapping range.
        """
        # copy the original pointcloud
        pcd_keypoints_local = open3d.geometry.PointCloud(pcd_keypoints)
        num_keypoints = np.asarray(pcd_keypoints_local.points).shape[0]
        idxs_all_keypoints = np.linspace(start=0, stop=num_keypoints - 1, num=num_keypoints, dtype=np.int32)
        # transform the keypoints into the map frame
        # this is done under the assumptions that, at some point, there are more landmarks than keypoints being observed
        pcd_keypoints_local = pcd_keypoints_local.transform(self.x.as_t3d())
        n_landmarks, *_ = self.landmarks.shape
        if n_landmarks == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32),  # closest correspondence
                idxs_all_keypoints,  # previously unseen correspondences
            )
        else:
            # create a KD-Tree for correspondence search
            kdtree_keypoints = open3d.geometry.KDTreeFlann(pcd_keypoints_local)
            correspondences = np.empty((0, 2), dtype=np.int32)
            c_distances = np.empty((0, 1), dtype=np.float64)
            idxs_outlier_keypoints = np.empty(0, dtype=np.int32)
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
                    idxs_outlier_keypoints = np.hstack((idxs_outlier_keypoints, np.delete(idxs, idx_min_dist)))
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
                non_outlier_unmatched_keypoints = np.setdiff1d(unmatched_keypoints, idxs_outlier_keypoints)
                idxs_matched_landmarks = correspondences[:, 0]
                num_landmarks = self.landmarks.shape[0]
                idxs_all_landmarks = np.linspace(start=0, stop=num_landmarks - 1, num=num_landmarks, dtype=np.int32)
                # TODO: can we assume unique? (see param "assume_unique"), in
                # https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html
                idxs_unmatched_landmarks = np.setdiff1d(idxs_all_landmarks, idxs_matched_landmarks)
                idxs_landmarks_in_range = self.__get_idxs_landmarks_in_range(max_distance=max_observation_range)
                # the indices of all unobserved landmarks that should have been observed
                idxs_unobserved_landmarks_in_range = np.intersect1d(idxs_unmatched_landmarks, idxs_landmarks_in_range)
                # increment the observation counter for matched landmarks
                self.observation_counter[idxs_matched_landmarks, 0] += 1
                # decrement the observation counter for unmatched landmarks
                self.observation_counter[idxs_unobserved_landmarks_in_range, 0] -= 1
                return (correspondences, closest_corresondence, non_outlier_unmatched_keypoints)

    def add_new_landmarks_from_keypoints(
        self,
        idxs_new_landmarks: npt.NDArray[np.int32],
        keypoints_in_robot_frame: npt.NDArray[np.float64],
        position_covariance: npt.NDArray[np.float64],
        max_active_landmarks: int = -1,
    ) -> None:
        """Add a subset of keypoints to the map if they have previously been unobserved.

        ### Remarks

        #### Initializing new Landmarks

        The array `idxs_new_landmarks` indicates which of the `keypoints_in_robot_frame` should be added to the map.
        To add new landmarks, the `keyponits_in_robot_frame` are first transformed into the map frame.
        The covariance of each new landmark is then set to `position_covariance`.

        #### Restricting the Map Size

        The amount of landmarks that will be added to the map is determined by the `max_active_landmarks` parameter.
        If `max_active_landmarks > 0`, only the first `max_active_landmarks - currently_active_landmarks` will be admitted to the map.

        ### Pruning spurios Landmarks

        Set `max_active_landmarks < 0` to allow an arbitrary amount of landmarks to be admitted to the map.
        If this is the case, the amount of landmarks within the map is then solely determined by the `prune_landmarks()` method.
        """
        # number of keypoints that can be admitted into the map
        num_new = max_active_landmarks - self.landmarks.shape[0]
        pcd_l_new = open3d.geometry.PointCloud(
            # use all keypoints if there is no restriction
            open3d.utility.Vector3dVector(keypoints_in_robot_frame[idxs_new_landmarks])
            if max_active_landmarks < 0
            # otherwise use as many as is allowed
            else open3d.utility.Vector3dVector(keypoints_in_robot_frame[idxs_new_landmarks][:num_new])
        )
        # transform the keyponits into the map frame
        pcd_l_new = pcd_l_new.transform(self.x.as_t3d())
        pcd_l_new_checked = self.__landmark_admission_check_filter(pcd_l_new)
        l_new = np.asarray(pcd_l_new_checked.points)
        self.__add_landmarks(ls=l_new, Q_0=position_covariance)
        self.landmarks_initialized = True

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
            # ... likely due to small numerical offsets that build up as time goes on
            C = np.linalg.cholesky(S_l)
            HC = H_l @ C
            Q_l = HC @ HC.T + Q_z
            L = np.linalg.cholesky(Q_l)
            L_inv = np.linalg.inv(L)
            Q_l_inv = L_inv @ L_inv.T
            # kalman gain
            K_l = S_l @ H_l.T @ Q_l_inv
            # update the landmark in question
            self.__update_landmark(idx_l, delta=delta_z, K_gain=K_l, H=H_l, Qz=Q_z)
            # set observation error and its covariance
            ds[idx_l] = delta_z
            Qs[idx_l] = Q_l
        # return the errors and covariances for likelihood computation
        return (ds, Qs)

    def prune_landmarks(self, max_unobserved_count: int = -1, max_distance: Optional[float] = None) -> None:
        """Removes landmarks that are unobserved too often (or - optionally - exceed the mapping range).

        This is basically a post-update map-management method.
        """
        num_landmarks = self.landmarks.shape[0]
        idxs_landmarks_unobserved_too_often = np.where(self.observation_counter <= max_unobserved_count)
        # if there is no pruning range supplied, keep all landmarks regardless of their distance
        idxs_landmarks_in_range = (
            self.__get_idxs_landmarks_in_range(max_distance)
            if max_distance
            else np.linspace(start=0, stop=num_landmarks - 1, num=num_landmarks, dtype=np.int32)
        )
        idxs_to_keep = np.setdiff1d(idxs_landmarks_in_range, idxs_landmarks_unobserved_too_often)
        # keep all the map-related values of the landmarks in range and aren't unobserved too often
        self.landmarks = self.landmarks[idxs_to_keep]
        self.landmark_covariances = self.landmark_covariances[idxs_to_keep]
        self.observation_counter = self.observation_counter[idxs_to_keep]

    def get_trajectory(self) -> np.ndarray:
        """Obtain the trajectory traversed over the lifetime of this particle."""
        return np.vstack((self.trajectory, self.x.as_twist()))

    def __get_idxs_landmarks_in_range(self, max_distance: float = 10.0) -> np.ndarray:
        robot_position = np.array([*self.x.as_twist()[:2], 0], dtype=np.float64)
        relative_landmark_positions = np.copy(self.landmarks) - robot_position
        landmark_distances = np.linalg.norm(relative_landmark_positions, axis=1)
        idxs_landmarks_in_range, *_ = np.where(landmark_distances <= max_distance)
        return idxs_landmarks_in_range

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

    def __landmark_admission_check_filter(
        self, pcd_candidates: open3d.geometry.PointCloud, min_mutual_distance: float = 3.0
    ) -> open3d.geometry.PointCloud:
        """Filter landmark candidates for those that pass the admission check.

        Current admission checks:
            - must have at least `min_mutual_distance` to every landmark in the map

        ### Remark

        Will passthrough the input if the map is empty.
        """
        # skip if there aren't any landmarks in the map
        if self.landmarks.shape[0] == 0:
            return pcd_candidates
        # ohterwise transform the keypoints into the map frame and perform checks
        pcd_landmarks = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(self.landmarks))
        kdtree_landmarks = open3d.geometry.KDTreeFlann(pcd_landmarks)
        pts_candidates = np.asarray(pcd_candidates.points, dtype=np.float64)
        passed = np.repeat(False, pts_candidates.shape[0])
        for idx_candidate, pt_candidate in enumerate(pts_candidates):
            num_matches, *_ = kdtree_landmarks.search_radius_vector_3d(query=pt_candidate, radius=min_mutual_distance)
            if num_matches == 0:
                passed[idx_candidate] = True
        idxs_passed, *_ = np.where(passed)
        pts_passed = pts_candidates[idxs_passed]
        return open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pts_passed))

    def _dbg_set_pose(self, pose: Pose2D) -> None:
        """A debugging method to directy set the particles pose (i.e. from ground truth)."""
        self.x = pose
