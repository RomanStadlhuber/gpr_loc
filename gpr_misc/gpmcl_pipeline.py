# from gpmcl.scan_tools_3d import ScanTools3D
from gpmcl.localization_scenario import (
    LocalizationScenario,
    LocalizationScenarioConfig,
    LocalizationSyncMessage,
    LocalizationPipeline,
)
from gpmcl.particle_filter import (
    ParticleFilter,
    ParticleFilterConfig,
)
from gpmcl.mapper import MapperConfig, Mapper
from gpmcl.scan_tools_3d import ScanTools3D, PointCloudVisualizer
from gpmcl.regression import GPRegression, GPRegressionConfig
from transform import odometry_msg_to_affine_transform
from typing import Optional, Dict
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

    def __init__(self, config: Dict, debug_visualize: bool = False) -> None:
        self.config = config  # dict containing the pipeline configuration
        self.debug_visualize = debug_visualize
        # the evaluation trajectories
        self.df_trajectory_estimated = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_trajectory_groundtruth = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_trajectory_odometry = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_particles = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_landmarks = pd.DataFrame(columns=["x", "y"])
        # a count used to print the number of iterations already performed by the filter
        self.debug_iteration_count = 0
        # keep a mapper for testing purposes
        mapper_config = self.__get_mapper_config(self.config)
        self.mapper = Mapper(mapper_config)
        self.visualizer = PointCloudVisualizer()

    def initialize(self, synced_msgs: LocalizationSyncMessage) -> None:
        GP_process = self.__get_process_gp(self.config)
        pf_config = self.__get_pf_config(self.config)
        # instantiate the particle filter
        self.pf = ParticleFilter(
            config=pf_config,
            process_regressor=GP_process,
            initial_ground_truth=synced_msgs.groundtruth,
            initial_odom_estimate=synced_msgs.odom_est,
        )

    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        pcd_scan = ScanTools3D.pointcloud2_to_open3d_pointcloud(synced_msgs.scan_3d)
        self.mapper.compute_scan_correspondences_to_map(pcd_scan)
        if synced_msgs.groundtruth is not None:
            T_curr = odometry_msg_to_affine_transform(synced_msgs.groundtruth)
            self.mapper.update_map(pose=T_curr)
            self.visualizer.update([self.mapper.pcd_map])
            map_points = np.asarray(self.mapper.pcd_map.points)
            print(f"Map contains {map_points.shape[0]} points.")

        else:
            return

        # region

        # increment the iteration counter
        self.debug_iteration_count += 1
        print(f"Iteration {self.debug_iteration_count}.")
        # actual inference begins here
        # pcd = ScanTools3D.scan_msg_to_open3d_pcd(synced_msgs.scan_3d)
        # compute the prior by sampling from the GP
        # self.pf.predict(odom=synced_msgs.odom_est)
        # self.pf.update(None)
        # print(f"[{timestamp}]: iteration {self.debug_iteration_count}, w_eff: {self.pf.M_eff/self.pf.M}")
        # # update the trajectory dataframe
        # self.__update_trajectory(
        #     # provide current estimate as twist (x, y, theta)
        #     estimate=self.pf.mean().as_twist(),
        #     # provide ground truth pose as twist (x, y, theta) if available
        #     groundtruth=Pose2D.from_odometry(synced_msgs.groundtruth).as_twist()
        #     if synced_msgs.groundtruth is not None
        #     else None,
        #     odometry=Pose2D.from_odometry(synced_msgs.odom_est).as_twist(),
        # )

        # endregion

    def export_trajectory(self, out_dir: pathlib.Path) -> None:
        # create output directory if it does not exist
        if not out_dir.exists():
            out_dir.mkdir()
        self.df_trajectory_estimated.to_csv(out_dir / "trajectory_estimated.csv")
        self.df_particles.to_csv(out_dir / "particles.csv")
        self.df_landmarks.to_csv(out_dir / "landmarks.csv")
        df_rows, *_ = self.df_trajectory_groundtruth.shape
        if df_rows > 0:
            self.df_trajectory_groundtruth.to_csv(out_dir / "trajectory_groundtruth.csv")
        df_rows, *_ = self.df_trajectory_odometry.shape
        if df_rows > 0:
            self.df_trajectory_odometry.to_csv(out_dir / "trajectory_odometry.csv")

    def __get_pf_config(self, config: Dict) -> ParticleFilterConfig:
        return ParticleFilterConfig.from_config(config)

    def __get_mapper_config(self, config: Dict) -> MapperConfig:
        return MapperConfig.from_config(config)

    def __get_process_gp(self, config: Dict) -> GPRegression:
        # load the process GP from the config
        gp_config = GPRegressionConfig.from_config(config=config, key="process_gp")
        gp = GPRegression(gp_config)
        return gp

    def __update_trajectory(
        self, estimate: np.ndarray, groundtruth: Optional[np.ndarray] = None, odometry: Optional[np.ndarray] = None
    ) -> None:
        """Update the dataframes containing the estimated and (optionally) ground truth trajectories."""
        # the current index is the length of the dataframe
        idx_trajectory_curr, *_ = self.df_trajectory_estimated.shape
        self.df_trajectory_estimated.loc[idx_trajectory_curr, :] = estimate
        partilces_weighted = np.hstack((self.pf.Xs, self.pf.ws.reshape(-1, 1)))
        df_particles_curr = pd.DataFrame(columns=["x", "y", "theta", "w"], data=partilces_weighted)
        self.df_particles = pd.concat((self.df_particles, df_particles_curr), ignore_index=True)
        # df_landmarks_curr = pd.DataFrame(columns=["x", "y"], data=self.pf.mapper.get_map().as_matrix()[:, :2])
        # self.df_landmarks = pd.concat((self.df_landmarks, df_landmarks_curr))
        # store ground truth if provided
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
    config_file = open(config_path)
    config = yaml.safe_load(config_file)
    dbg_vis = args.debug_visualize
    out_dir = pathlib.Path(args.out_dir)
    # instantiate the pipeline
    pipeline = GPMCLPipeline(config, debug_visualize=dbg_vis)
    # configure the scenario
    scenario_config = LocalizationScenarioConfig.from_config(config)
    # instantiate the scenario
    localization_scenario = LocalizationScenario(config=scenario_config, pipeline=pipeline)
    # run localization inference
    localization_scenario.spin_bag()
    localization_scenario.export_metrics(out_dir)
