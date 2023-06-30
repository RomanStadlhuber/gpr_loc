# TODO: re-implement the mapper using
# - only open3d PCDs to represent landmarks
# - KD tree search to find correspondences
from dataclasses import dataclass
from typing import Dict, Optional
from scipy.spatial.transform import Rotation
import numpy as np
import open3d

@dataclass
class MapperConfig:
    downsampling_voxel_size: float = 0.05
    scan_tf: np.ndarray = np.eye(4, dtype=np.float64)

    @staticmethod
    def from_config(config:Dict) -> "MapperConfig":
        scan_tf: Optional[Dict] = config.get("scan_tf")
        if scan_tf is not None:
            downsampling_voxel_size:Optional[float] = config.get("downsampling_voxel_size")
            # --- load scan TF
            # region
            scan_translation = np.array(scan_tf.get("position"), dtype=np.float64)
            scan_orientation = np.array(scan_tf.get("orientation"), dtype=np.float64)
            R_scan = Rotation.from_quat(scan_orientation).as_matrix()
            t_scan = scan_translation.reshape((3,1))
            T_scan = np.block([ # type: ignore
                [R_scan, t_scan],
                [0,0,0,1]
            ], dtpye=np.float64)
            # endregion
            return MapperConfig(scan_tf=T_scan, downsampling_voxel_size=downsampling_voxel_size or 0.05,)
        else: 
            return MapperConfig()



class Mapper:
    def __init__(self, config: MapperConfig) -> None:
        self.config = config
        # map starts out as empty point cloud
        self.pcd_map = open3d.geometry.PointCloud()
        # the features for the map are empty also
        self.features_map = open3d.pipelines.registration.Feature()
        pass

    def process_scan(self, pcd: open3d.geometry.PointCloud) -> None:
        """Compute features for a scan and store it for the next iteration."""
        # downsample the scan
        # pcd = pcd.voxel_down_sample(voxel_size=self.config.downsampling_voxel_size)
        # transform the scan into the base frame
        pcd = pcd.transform(self.config.scan_tf)
        # estimate normals using the default settings
        pcd.estimate_normals()
        # compute the FPFH features for the downsampled scan
        features = open3d.pipelines.registration.compute_fpfh_feature(
            input=pcd,
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=100)
        )
        if self.pcd_map.is_empty():
            self.pcd_map = pcd
            self.features_map = features
        else:
            # TODO: perform feature correspondence estimation and return
            # landmarks and observed features for the PFs likelihood computation
            pass

