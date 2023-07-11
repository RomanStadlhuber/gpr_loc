from gpmcl.localization_scenario import (
    LocalizationScenario,
    LocalizationSyncMessage,
    LocalizationPipeline,
)
from gpmcl.mapper import Mapper
from gpmcl.scan_tools_3d import ScanTools3D, PointCloudVisualizer
from gpmcl.config import load_gpmapping_offline_config, GPMappingOfflineConfig
from gpmcl.transform import odometry_msg_to_affine_transform, Pose2D
from gpmcl.motion_model import MotionModel
from gpmcl.fast_slam import FastSLAM
from typing import Optional
import numpy as np
import pandas as pd
import argparse
import pathlib

# TODO:
# the pipeline will later use the following modules
# - Odometry Module (which will use GP inference)
# - Mapping Module (instead of ScanTools3D, which will be internal)
# - Filtering Module (which will glue Odometry and Mapping Together)


class GPMCLPipeline(LocalizationPipeline):
    """Pipeline for Gaussian Process Monte Carlo Localization

    Uses a Gaussian Process Particle Filter (GP-PF).
    """

    def __init__(self, config: GPMappingOfflineConfig, debug_visualize: bool = False) -> None:
        self.config = config  # dict containing the pipeline configuration
        self.debug_visualize = debug_visualize
        # the evaluation trajectories
        self.df_trajectory_estimated = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_trajectory_groundtruth = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_trajectory_odometry = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_particles = pd.DataFrame(columns=["x", "y", "theta"])
        # a count used to print the number of iterations already performed by the filter
        self.debug_iteration_count = 0
        # the visualizer used to display the 3D scan map
        self.visualizer = PointCloudVisualizer()
        # the 3D scan mapper
        self.mapper = Mapper(self.config["mapper"])
        # motion model used for GP based Fast SLAM
        self.motion_model = MotionModel(config=self.config["motion_model_gp"])
        # GP based Fast SLAM
        self.slam = FastSLAM(config=self.config["fast_slam"], motion_model=self.motion_model)

    def initialize(self, synced_msgs: LocalizationSyncMessage) -> None:
        # sample initial guess about ground truth or first odometry estimate
        initial_guess_pose = Pose2D.from_odometry(synced_msgs.groundtruth or synced_msgs.odom_est)
        self.slam.initialize_from_pose(x0=initial_guess_pose.as_twist())
        # buffer the latest estimated pose, this is used to compute estimated motion from odometry
        self.odom_last = Pose2D.from_odometry(synced_msgs.odom_est)

    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        pcd_scan = ScanTools3D.pointcloud2_to_open3d_pointcloud(synced_msgs.scan_3d)
        pcd_keypoints = self.mapper.process_scan(pcd_scan)
        odom_curr = Pose2D.from_odometry(synced_msgs.odom_est)
        delta_odom = Pose2D.delta(self.odom_last, odom_curr)
        self.slam.predict(estimated_motion=delta_odom)
        self.slam.update(keypoints=pcd_keypoints)
        # if synced_msgs.groundtruth is not None:
        #     T_curr = odometry_msg_to_affine_transform(synced_msgs.groundtruth)
        #     self.mapper.update_map(pose=T_curr)
        #     self.visualizer.update([self.mapper.pcd_map])
        #     map_points = np.asarray(self.mapper.pcd_map.points)
        #     print(f"Map contains {map_points.shape[0]} points.")
        # increment the iteration counter
        self.debug_iteration_count += 1
        print(f"Iteration {self.debug_iteration_count}.")
        self.odom_last = odom_curr
        self.__update_trajectory()

    def export_trajectory(self, out_dir: pathlib.Path) -> None:
        # create output directory if it does not exist
        if not out_dir.exists():
            out_dir.mkdir()
        self.df_trajectory_estimated.to_csv(out_dir / "trajectory_estimated.csv")
        self.df_particles.to_csv(out_dir / "particles.csv")
        df_rows, *_ = self.df_trajectory_groundtruth.shape
        if df_rows > 0:
            self.df_trajectory_groundtruth.to_csv(out_dir / "trajectory_groundtruth.csv")
        df_rows, *_ = self.df_trajectory_odometry.shape
        if df_rows > 0:
            self.df_trajectory_odometry.to_csv(out_dir / "trajectory_odometry.csv")

    def __update_trajectory(
        self, estimate: np.ndarray, groundtruth: Optional[np.ndarray] = None, odometry: Optional[np.ndarray] = None
    ) -> None:
        """Update the dataframes containing the estimated and (optionally) ground truth trajectories."""
        # the current index is the length of the dataframe
        idx_trajectory_curr, *_ = self.df_trajectory_estimated.shape
        self.df_trajectory_estimated.loc[idx_trajectory_curr, :] = estimate
        particles = self.slam.get_particle_poses()
        df_particles_curr = pd.DataFrame(columns=["x", "y", "theta"], data=particles)
        self.df_particles = pd.concat((self.df_particles, df_particles_curr), ignore_index=True)
        if groundtruth is not None:
            self.df_trajectory_groundtruth.loc[idx_trajectory_curr, :] = groundtruth
        if odometry is not None:
            self.df_trajectory_odometry.loc[idx_trajectory_curr, :] = odometry


arg_parser = argparse.ArgumentParser(
    prog="gpmcl_pipeline", description="Perform localization using gaussian process regression"
)

arg_parser.add_argument(
    "config_path",
    metavar="Path to the configuration YAML file.",
    type=str,
)

arg_parser.add_argument(
    "-d",
    "--dbg_vis",
    dest="debug_visualize",
    required=False,
    # metavar="Visualize the point cloud, features and landmarks for debugging.",
    # help="Set this flag if the map and correspondences + outliers should be visualized at each step.",
    default=False,
    action="store_true",
)

arg_parser.add_argument(
    "-o",
    "--out_dir",
    dest="out_dir",
    metavar="out_dir",
    type=str,
    help="Evaluation metrics export directory.",
    default=".",
)

if __name__ == "__main__":
    args = arg_parser.parse_args()
    config_path = pathlib.Path(args.config_path)
    gpmapping_config = load_gpmapping_offline_config(config_path)
    dbg_vis = args.debug_visualize
    out_dir = pathlib.Path(args.out_dir)
    # instantiate the pipeline
    pipeline = GPMCLPipeline(config=gpmapping_config, debug_visualize=dbg_vis)
    # instantiate the scenario
    localization_scenario = LocalizationScenario(config=gpmapping_config["bag_runner"], pipeline=pipeline)
    # run localization inference
    localization_scenario.spin_bag()
    # TODO: re-enable this once the divergence is fixed
    # localization_scenario.export_metrics(out_dir)
