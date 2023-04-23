from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import open3d


@dataclass
class Feature3D:
    """A wrapper for representation of arbitrary 3D features.

    Allows conversion between `np.ndarray` and `Feature3D` instances.
    """

    position: np.ndarray  # (x, y, z) position of the feature

    def as_vector(self) -> np.ndarray:
        return np.array(self.position)

    def covariance(self) -> np.ndarray:
        """Get the covariance matrix associated with a feature."""
        # TODO: set a better covariance function based on the feature position
        dim_f, *_ = np.shape(self.position)
        dim_sig = dim_f - 3
        # assumes: covariance is zero for all signatures
        return np.block(
            [
                [np.eye(3), np.zeros((3, dim_sig))],
                [np.zeros((dim_sig, 3)), np.zeros((dim_sig, dim_sig))],
            ]
        )

    @staticmethod
    def from_vector(f: np.ndarray) -> "Feature3D":
        return Feature3D(position=f)


@dataclass
class FeatureToLandmarkCorrespondence3D:
    """Unambiguous wrapper for feature-to-landmark correspondences.

    Use this instead of a tuple to keep things clear.
    Also, this does not use the features themselves as they might be in different frames
    """

    idx_landmark: int
    idx_feature: int


@dataclass
class CorrespondenceSearchResults3D:
    """Contains both the matched correspondencies as well as outliers."""

    landmark_outlier_idxs: List[int]
    feature_outlier_idxs: List[int]
    correspondences: List[FeatureToLandmarkCorrespondence3D]


@dataclass
class FeatureMap3D:
    """A wrapper for representation of 3D feature maps.

    Allows converstion between `np.ndarray` and `FeatureMap3D`.
    """

    # the 3D feature vectors
    features: List[Feature3D]
    # the frame in which the features are captured
    frame: Optional[str] = None

    def at(self, idx: int) -> Feature3D:
        """Get the feature at the desired index"""
        return self.features[idx]

    def as_matrix(self) -> np.ndarray:
        """Convert the features to a `(DIM_F x N)` map matrix"""
        feature_vectors = list(map(lambda f: f.as_vector(), self.features))
        return np.array(feature_vectors)

    def transform(self, T: np.ndarray) -> "FeatureMap3D":
        """Generate a new - transformed - feature map based on the current one.

        NOTE: does not do an in-place update because this shouldn't be allowed on a global map.
        """
        # get feature matrix
        M = self.as_matrix().T  # transpose to convert to (dim_f x N)

        R = T[:3, :3]  # get rotation matrix
        t = T[:3, 3]  # get translation vector
        dim_f, num_f = np.shape(M)  # get dimension and number of features
        dim_s = dim_f - 3  # get the dimension of signature

        # transformation matrix of feature position and signature
        # in homogeneous space, this matrix transforms all positions but keeps the signatures constant
        A = np.block(
            [
                [R, np.zeros((3, dim_s)), t.reshape(-1, 1)],
                [np.zeros((dim_s, 3)), np.eye(dim_s), np.zeros((dim_s, 1))],
            ]
        )
        # feature matrix in homogeneous coordinates
        M_h = np.vstack((M, np.ones((1, num_f), dtype=np.float64)))
        # transform in homegeneous space and extract features
        M = (A @ M_h)[:dim_f, :].T  # transpone to convert to (N x dim_f)

        return FeatureMap3D.from_matrix(M)

    @staticmethod
    def from_matrix(M: np.ndarray) -> "FeatureMap3D":
        feature_vectors = M.tolist()  # convert to regular python list
        # apply interpretation mapping
        features = list(map(lambda fvec: Feature3D.from_vector(fvec), feature_vectors))
        # convert back to Map wrapper
        return FeatureMap3D(features)

    @staticmethod
    def from_pcd(pcd: open3d.geometry.PointCloud) -> "FeatureMap3D":
        M = np.asarray(pcd.points, dtype=np.float64)
        feature_map = FeatureMap3D.from_matrix(M)
        return feature_map


class Mapper(ABC):
    """The abstract base class for different feature based mapping modules"""

    @abstractmethod
    def detect_features(self, pcd: open3d.geometry.PointCloud) -> FeatureMap3D:
        """Detect 3D features as they are used by the mapper implementation."""
        pass

    @abstractmethod
    def correspondence_search(
        self, observed_features: FeatureMap3D, pose: np.ndarray
    ) -> CorrespondenceSearchResults3D:
        """Perform correspondence search between the observed features and the global map landmarks."""
        pass

    @abstractmethod
    def update(
        self,
        observed_features: FeatureMap3D,
        pose: np.ndarray,
        correspondences: CorrespondenceSearchResults3D,
    ) -> None:
        """Update the map based on the mappers internal criteria."""
        pass


@dataclass
class ISS3DMapperConfig:
    """Configuration wrapper for the ISS3D Mapper class.

    Contains information such as the mapping radius and saliency criteria.

    Also allos to set a non-maxima suppression radius.
    However, this heavily reduces the number of detected features and setting
    a low radius is encouraged.
    Setting `nms_radius` to `0.0` disables NMS alltogether.
    """

    # voxel size used when downsampling the pcd
    downsampling_voxel_size: float = 0.15
    # the radius to consider features
    mapping_radius: float = 12.0  # [m]
    # non-maxima suppression radius (zero means no NMS is applied)
    nms_radius: float = 1.5  # [m]
    # max ratio between eigenvalues 1 and 2 (lower is more conservative)
    ratio_eigs_1_2: float = 0.65
    # max ratio between eigenvalues 2 and 3 (lower is more conservative)
    ratio_eigs_2_3: float = 0.65
    # minumum number of supporting points to consider a feature
    min_neighbor_count: int = 5


class ISS3DMapper(Mapper):
    """Mapper based on 3D Intrinsic Shape Signatures."""

    def __init__(
        self,
        config: Optional[ISS3DMapperConfig] = None,
        initial_guess_pose: np.ndarray = np.eye(4),
    ) -> None:
        # use default configuration if none was provided
        self.config = config or ISS3DMapperConfig()
        # initialize the global map
        self.map = FeatureMap3D(features=[], frame="map")
        # set the initial guess pose (required for global map)
        self.T0 = initial_guess_pose

    def detect_features(self, pcd: open3d.geometry.PointCloud) -> FeatureMap3D:
        """Detect and filter ISS3D features in the input point cloud."""
        # downsampling to obtain more robust features
        downsampled_pcd = pcd.voxel_down_sample(
            voxel_size=self.config.downsampling_voxel_size
        )
        # obtain a pointcloud with the detected features
        pcd_keypoints = open3d.geometry.keypoint.compute_iss_keypoints(
            input=downsampled_pcd,
            salient_radius=self.config.mapping_radius,
            non_max_radius=self.config.nms_radius,
            gamma_21=self.config.ratio_eigs_1_2,
            gamma_32=self.config.ratio_eigs_2_3,
            min_neighbors=self.config.min_neighbor_count,
        )
        return FeatureMap3D.from_pcd(pcd_keypoints)

    def correspondence_search(
        self, observed_features: FeatureMap3D, pose: np.ndarray
    ) -> CorrespondenceSearchResults3D:
        """Perform correspondence search at the current frame"""

        if len(self.map.features) == 0:
            return CorrespondenceSearchResults3D(
                feature_outlier_idxs=list(range(len(observed_features.features))),
                landmark_outlier_idxs=[],
                correspondences=[],
            )

        # TODO: implement mahalanobis distance minimizer search
        R = pose[:3, :3]
        t = pose[:3, 3].reshape(-1, 1)  # reshape to column vector i.e. (3x1)
        # invert the pose
        T_inv = np.block(
            [
                [R.T, R.T @ -t],
                [np.zeros((1, 3)), 1.0],
            ]
        )
        # transform the map into the local frame
        # NOTE: this turns the landmarks into observations!
        local_map = self.map.transform(T_inv)
        # block matrix containing the estimated observations
        Z_est = local_map.as_matrix()
        n_landmarks, *_ = np.shape(Z_est)
        n_features = len(observed_features.features)
        # book-keeping vector for correspondences
        # 0 means "no correspondence yet"
        # k > 0 means "a correspondence to feature k'
        # -1 means "invalid due to ambiguity"
        Cs = -1 * np.ones((n_landmarks))

        # find correspondences for each observation
        for k, f_k in enumerate(observed_features.features):

            def mahalanobis_distance(estimated_feature: Feature3D) -> float:
                """Compute the mahalanobis distance of an estimated feature to the current observation."""
                z_k = f_k.as_vector()
                z_est = estimated_feature.as_vector()
                S = estimated_feature.covariance()
                delta = z_k - z_est
                return delta.T @ S @ delta

            deltas = np.array(list(map(mahalanobis_distance, local_map.features)))
            # correspondence for feature k given all landmarks
            c_k = np.argmin(deltas)
            # set all ambiguous landmarks to be invalid matches
            Cs = np.where(Cs == k, -2, Cs)
            # set the landmark at index "c_k" to correspond to feature k
            if not Cs[c_k] == -2:
                Cs[c_k] = k

        # type cast book-keeping array to integers
        Cs = Cs.astype("int8")
        # generate correspondences from bookkeeping vector
        correspondences = [FeatureToLandmarkCorrespondence3D(idx_landmark=i, idx_feature=k) for i, k in enumerate(Cs) if k >= 0]
        # all indices of unmatched or ambiguous landmarks
        idxs_landmark_outliers = [i for i in range(n_landmarks) if Cs[i] < 0]
        # to find the outliers, we must fist find the inliers
        idxs_feature_inliers = [k for k in range(n_landmarks) if Cs[k] >= 0]
        # all indices of unmatched or ambigous features
        idxs_feature_outliers = [
            k for k in range(n_features) if k not in idxs_feature_inliers
        ]

        # pack the resutls into the wrapper
        search_results = CorrespondenceSearchResults3D(
            landmark_outlier_idxs=idxs_landmark_outliers,
            feature_outlier_idxs=idxs_feature_outliers,
            correspondences=correspondences,
        )

        return search_results

    def update(
        self,
        observed_features: FeatureMap3D,
        pose: np.ndarray,
        correspondences: CorrespondenceSearchResults3D,
    ) -> None:
        # at first, all features need to be transformed into the map frame
        features_in_map_frame = observed_features.transform(pose)

        # --- if the map is empty, add all features ---
        # region
        if len(self.map.features) == 0:
            self.map.features = features_in_map_frame.features
            return
        # endregion

        # TODO: implement an N-times unobserved tracker for landmarks!
        # --- only keep landmarks that have been re-matched ---
        # region
        inlier_features = [
            feature
            for idx, feature in enumerate(self.map.features)
            if idx not in correspondences.landmark_outlier_idxs
        ]
        self.map.features = inlier_features
        # endregion

        # --- add all previously unseen features ---
        # region
        # keep all observed features that were previously unseen i.e. not in the map
        features_previously_unseen = [
            feature
            for idx, feature in enumerate(features_in_map_frame.features)
            if idx in correspondences.feature_outlier_idxs
        ]
        self.map.features.extend(features_previously_unseen)
        # endregion

        # --- keep only features within the mapping range ---
        # region
        pose_position = pose[:3, 3]
        # keep only features that are within bounds
        features_in_bounds = [
            feature
            for feature in self.map.features
            if np.linalg.norm(feature.position - pose_position)
            <= self.config.mapping_radius
        ]
        self.map.features = features_in_bounds
        # endregion

        return

    def __filter_for_LOAM_features(self, features: FeatureMap3D) -> FeatureMap3D:
        """Filter a feature map according to LOAM conventions.

        That is, divide the map into quadrants and only select the six most salient features.
        """

        # TODO: apply LOAM filtering
        return features
