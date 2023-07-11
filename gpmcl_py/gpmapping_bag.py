# from gpmcl.scan_tools_3d import ScanTools3D
from gpmcl.localization_scenario import (
    LocalizationScenario,
    LocalizationSyncMessage,
    LocalizationPipeline,
)
from gpmcl.mapper import Mapper
from gpmcl.scan_tools_3d import ScanTools3D, PointCloudVisualizer
from gpmcl.config import load_gpmapping_offline_config, GPMappingOfflineConfig
from gpmcl.transform import odometry_msg_to_affine_transform
from typing import Optional
import numpy as np
import pandas as pd
import argparse
import pathlib
import yaml

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
        self.mapper = Mapper(self.config["mapper"])
        self.visualizer = PointCloudVisualizer()

    def initialize(self, synced_msgs: LocalizationSyncMessage) -> None:
        pass

    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        pcd_scan = ScanTools3D.pointcloud2_to_open3d_pointcloud(synced_msgs.scan_3d)
        pcd_keypoints = self.mapper.process_scan(pcd_scan)
        if synced_msgs.groundtruth is not None:
            T_curr = odometry_msg_to_affine_transform(synced_msgs.groundtruth)
            self.mapper.update_map(pose=T_curr)
            self.visualizer.update([self.mapper.pcd_map])
            map_points = np.asarray(self.mapper.pcd_map.points)
            print(f"Map contains {map_points.shape[0]} points.")

        else:
            return

        # increment the iteration counter
        self.debug_iteration_count += 1
        print(f"Iteration {self.debug_iteration_count}.")

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
        # region: store particle poses
        # TODO: implement a method to load particle poses from the filter
        # partilces_weighted = np.hstack((self.pf.Xs, self.pf.ws.reshape(-1, 1)))
        # df_particles_curr = pd.DataFrame(columns=["x", "y", "theta", "w"], data=partilces_weighted)
        # self.df_particles = pd.concat((self.df_particles, df_particles_curr), ignore_index=True)
        # endregion
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
