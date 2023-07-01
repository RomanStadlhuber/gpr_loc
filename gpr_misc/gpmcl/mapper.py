# TODO: re-implement the mapper using
# - only open3d PCDs to represent landmarks
# - KD tree search to find correspondences
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from scipy.spatial.transform import Rotation
import numpy as np
import open3d


@dataclass
class MapperConfig:
    downsampling_voxel_size: float = 0.05
    scan_tf: np.ndarray = np.eye(4, dtype=np.float64)

    @staticmethod
    def from_config(config: Dict) -> "MapperConfig":
        scan_tf: Optional[Dict] = config.get("scan_tf")
        if scan_tf is not None:
            downsampling_voxel_size: Optional[float] = config.get("downsampling_voxel_size")
            # --- load scan TF
            # region
            scan_translation = np.array(scan_tf.get("position"), dtype=np.float64)
            scan_orientation = np.array(scan_tf.get("orientation"), dtype=np.float64)
            R_scan = Rotation.from_quat(scan_orientation).as_matrix()
            t_scan = scan_translation.reshape((3, 1))
            T_scan = np.block([[R_scan, t_scan], [0, 0, 0, 1]], dtpye=np.float64)  # type: ignore
            # endregion
            return MapperConfig(
                scan_tf=T_scan,
                downsampling_voxel_size=downsampling_voxel_size or 0.05,
            )
        else:
            return MapperConfig()


class Mapper:
    def __init__(self, config: MapperConfig) -> None:
        self.config = config
        # the radius used for normal estimation
        self.radius_normal = config.downsampling_voxel_size * 2
        # the common configuration for estimating point normals
        self.normal_est_search_param = open3d.geometry.KDTreeSearchParamHybrid(
            radius=self.radius_normal,
            max_nn=30,
        )
        # the common configuration for estimating point cloud features
        self.feature_comp_search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=100)
        # map starts out as empty point cloud
        self.pcd_map = open3d.geometry.PointCloud()
        # there also need to be buffers for the current scan and its features
        self.pcd_scan_last = self.pcd_scan = open3d.geometry.PointCloud()
        self.features_scan_last = self.features_scan = open3d.pipelines.registration.Feature()
        # the set of correspondences between the current scan and the map
        self.correspondences = open3d.utility.Vector2iVector()
        pass

    def compute_scan_correspondences_to_map(
        self, pcd_scan_curr: open3d.geometry.PointCloud
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute features for a scan and store it for the next iteration."""
        # preprocess the scan and compute its features
        # region
        # downsample the scan
        self.pcd_scan = pcd_scan_curr.voxel_down_sample(voxel_size=self.config.downsampling_voxel_size)
        # remove reference to the input parameter to prevent incorrect access at a later time
        del pcd_scan_curr
        # transform the scan into the base frame
        self.pcd_scan.transform(self.config.scan_tf)
        # estimate normals using the default settings
        self.pcd_scan.estimate_normals(self.normal_est_search_param)
        # compute the FPFH features for the downsampled scan
        self.features_scan = open3d.pipelines.registration.compute_fpfh_feature(
            input=self.pcd_scan, search_param=self.feature_comp_search_param
        )
        # endregion
        if self.pcd_map.is_empty() and self.pcd_scan_last.is_empty():
            return []
        else:
            # compute mutual correspondences betweem the map and current scan
            self.correspondences = open3d.pipelines.registration.correspondences_from_features(
                source_features=self.features_scan,
                target_features=self.features_scan_last,
                mutual_filter=True,
            )
            correspondence_idxs = np.asarray(self.correspondences)
            # PCD points need to be cast from C++ vectors into numpy arrays first
            points_scan_last = np.asarray(self.pcd_scan_last.points, dtype=np.float64)
            points_scan = np.asarray(self.pcd_scan.points, dtype=np.float64)
            # list of corresponding (feature_position, landmark_position) 3D-vectors
            corresponding_features_and_landmarks = list(
                map(lambda c_i_j: (points_scan[c_i_j[0]], points_scan_last[c_i_j[1]]), correspondence_idxs)
            )
            return corresponding_features_and_landmarks

    def update_map(self, pose: np.ndarray):
        # transform the scan PCD into the current pose (needed for both initialization and update)
        # this assumes that the Scan-TF has already been applied
        # NOTE: transforming is not an in-place operation (very unfortunate..)
        self.pcd_scan.transform(pose)
        # here we assume that the pose argument represents the current pose of the scan in the map frame
        # mere the current scan into the map
        self.pcd_map += self.pcd_scan
        # downsample the merged PCDs to remove duplicate points
        # (see: http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html#Make-a-combined-point-cloud)
        self.pcd_map = self.pcd_map.voxel_down_sample(voxel_size=self.config.downsampling_voxel_size)
        # buffer the last scan for feature computation and matching in next iteration
        self.pcd_scan_last = self.pcd_scan
        # self.pcd_scan_last.estimate_normals(self.normal_est_search_param)
        # compute features for the last scan
        self.features_scan_last = open3d.pipelines.registration.compute_fpfh_feature(
            input=self.pcd_scan_last, search_param=self.feature_comp_search_param
        )
        # reset the correspondences
        self.correspondences = open3d.utility.Vector2iVector()
        # TODO: find out if this is really necessary
        # reset the current scan buffer and its features
        self.pcd_scan = open3d.geometry.PointCloud()
        self.features_scan = open3d.pipelines.registration.Feature()
