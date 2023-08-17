from typing import TypedDict, List, Optional
import pathlib
import yaml


class ScanTFConfig(TypedDict):
    position: List[float]
    orientation: List[float]


class MotionComponentLabelConfig(TypedDict):
    x: str
    y: str
    theta: str


class MotionModelGPConfig(TypedDict):
    model_dir: str
    estimated_motion_labels: MotionComponentLabelConfig
    estimated_twist_labels: MotionComponentLabelConfig


class MapperConfig(TypedDict):
    # the following parameters are required for keypoint detection
    scatter_radius: float
    nms_radius: float
    eig_ratio_32: float
    eig_ratio_21: float
    min_neighbor_count: int
    voxel_size: float
    scan_tf: ScanTFConfig
    min_height: float
    max_height: float


class FastSLAMConfig(TypedDict):
    # ignore features and landmarks beyond this range
    max_feature_range: float  # the max distance in which features are observed
    # these parameters are related to Fast SLAM
    max_unobserved_count: int
    particle_count: int
    keypoint_covariance: List[int]  # covariance of an XYZ keypoint
    observation_covariance: List[int]  # signal cov. of range-bearing observations
    # parameter for correspondence estimation using KDTree-FLANN
    kdtree_search_radius: float  # the nearest neighbor search radius
    motion_noise_gain: List[float]  # amplification of the motion noise to achieve higher spread


class BagRunnerConfig(TypedDict):
    bag_path: str
    estimated_odometry_topic: str
    pointcloud_topic: str
    groundtruth_topic: Optional[str]
    sync_period: float
    stop_after: Optional[int]


class GPMappingOfflineConfig(TypedDict):
    mapper: MapperConfig
    fast_slam: FastSLAMConfig
    motion_model_gp: MotionModelGPConfig
    bag_runner: BagRunnerConfig


def load_gpmapping_offline_config(config_path: pathlib.Path) -> GPMappingOfflineConfig:
    with open(config_path, "r") as f_conf:
        conf: GPMappingOfflineConfig = yaml.safe_load(stream=f_conf)
        return conf
