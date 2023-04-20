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
        return self.position

    def covariance(self) -> np.ndarray:
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
        self, observed_features: FeatureMap3D
    ) -> CorrespondenceSearchResults3D:
        """Perform correspondence search between the observed features and the global map landmarks."""
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

    # the radius to consider features
    mapping_radius: float = 12.0  # [m]
    # non-maxima suppression radius (zero means no NMS is applied)
    nms_radius: float = 2.5  # [m]
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
        # obtain a pointcloud with the detected features
        pcd_keypoints = open3d.geometry.keypoint.compute_iss_keypoints(
            input=pcd,
            salient_radius=self.config.mapping_radius,
            non_max_radius=self.config.nms_radius,
            gamma_21=self.config.ratio_eigs_1_2,
            gamma_32=self.config.ratio_eigs_2_3,
            min_neighbors=self.config.min_neighbor_count,
        )
        return FeatureMap3D.from_pcd(pcd_keypoints)

    def correspondence_search(self, observed_features: FeatureMap3D):
        pass

    def __filter_for_LOAM_features(self, features: FeatureMap3D) -> FeatureMap3D:
        """Filter a feature map according to LOAM conventions.

        That is, divide the map into quadrants and only select the six most salient features.
        """

        # TODO: apply LOAM filtering
        return features
