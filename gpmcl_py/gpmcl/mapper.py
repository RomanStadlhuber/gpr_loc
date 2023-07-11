# TODO: re-implement the mapper using
# - only open3d PCDs to represent landmarks
# - KD tree search to find correspondences
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from gpmcl.config import MapperConfig
import numpy as np
import open3d


class Mapper:
    def __init__(self, config: MapperConfig) -> None:
        self.config = config
        # the radius used for normal estimation
        self.radius_normal = 1.0
        # the common configuration for estimating point normals
        self.normal_est_search_param = open3d.geometry.KDTreeSearchParamHybrid(
            radius=self.radius_normal,
            max_nn=30,
        )
        # the common configuration for estimating point cloud features
        self.feature_comp_search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=100)
        # map starts out as empty point cloud
        self.pcd_map = open3d.geometry.PointCloud()
        # 3D affine transform from the base to the scan frame
        self.tf_scan = np.eye(4, dtype=np.float64)
        # set the scan TF translation
        self.tf_scan[:3, 3] = np.array(self.config["scan_tf"]["position"])
        # set the scan TF orientation
        Rmat_scan = Rotation.from_quat(np.array(self.config["scan_tf"]["orientation"])).as_matrix()
        self.tf_scan[:3, :3] = Rmat_scan

    def process_scan(self, pcd_scan_curr: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
        """Process the current scans pointcloud and compute ISS3D features"""
        # preprocess the scan and compute its keypoints
        # downsample the scan
        self.pcd_scan = pcd_scan_curr.voxel_down_sample(voxel_size=self.config["voxel_size"])
        # remove reference to the input parameter to prevent incorrect access at a later time
        del pcd_scan_curr
        # transform the scan into the base frame
        self.pcd_scan.transform(self.tf_scan)
        # estimate normals using the default settings
        self.pcd_scan.estimate_normals(self.normal_est_search_param)
        # compute ISS-3D Keypoints
        # see: http://www.open3d.org/docs/latest/tutorial/geometry/iss_keypoint_detector.html
        # and:
        # >>> help(open3d.geometry.keypoint.compute_iss_keypoints)
        pcd_keypoints = open3d.geometry.keypoint.compute_iss_keypoints(
            input=self.pcd_scan,
            salient_radius=self.config["scatter_radius"],
            non_max_radius=self.config["nms_radius"],
            gamma_21=self.config["eig_ratio_21"],
            gamma_32=self.config["eig_ratio_32"],
            min_neighbors=self.config["min_neighbor_count"],
        )
        return pcd_keypoints

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
        self.pcd_map = self.pcd_map.voxel_down_sample(voxel_size=self.config["voxel_size"])
        # further reduce the map point size by removing duplicated and invalid points
        self.pcd_map.remove_duplicated_points()
        self.pcd_map.remove_non_finite_points()
        # reset the correspondences
        self.correspondences = open3d.utility.Vector2iVector()

    # region: old registration methods
    # NOTE: these were used to compute common correspondences for all particles
    # however, since Fast-SLAM allows tracking different correspondences,
    # these are no longer needed.
    # NOTE: RANSAC with a proper initial guess was the most accurate method of all..

    # def __registration_FGR(self) -> open3d.pipelines.registration.RegistrationResult:
    #     """Register scans using fast global registration (FGR).
    #     see
    #     [open3d.pipelines.registration.registration_fgr_based_on_feature_matching](http://www.open3d.org/docs/latest/python_api/open3d.pipelines.registration.registration_ransac_based_on_feature_matching.html#open3d-pipelines-registration-registration-ransac-based-on-feature-matching)
    #     """
    #     reg_res = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #         source=self.pcd_scan,
    #         target=self.pcd_scan_last,
    #         source_feature=self.features_scan,
    #         target_feature=self.features_scan_last,
    #     )
    #     return reg_res

    # def __registration_RANSAC(self) -> open3d.pipelines.registration.RegistrationResult:
    #     """Register Scans using RANSAC.
    #     see
    #     [open3d.pipelines.registration.registration_ransac_based_on_feature_matching](http://www.open3d.org/docs/latest/python_api/open3d.pipelines.registration.registration_ransac_based_on_feature_matching.html#open3d-pipelines-registration-registration-ransac-based-on-feature-matching)
    #     """
    #     reg_res = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #         source=self.pcd_scan,
    #         target=self.pcd_scan_last,
    #         source_feature=self.features_scan,
    #         target_feature=self.features_scan_last,
    #         mutual_filter=True,
    #         max_correspondence_distance=0.5,
    #     )
    #     return reg_res

    # def __registration_ICP_Point2Plane(self) -> open3d.pipelines.registration.RegistrationResult:
    #     """Register scans using Point to Plane ICP."""
    #     reg_res = open3d.pipelines.registration.registration_icp(
    #         source=self.pcd_scan,
    #         target=self.pcd_scan_last,
    #         max_correspondence_distance=0.2,
    #         init=np.eye(4, dtype=np.float32),
    #         estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     )
    #     return reg_res

    # endregion
