from gpmcl.scan_tools_3d import ScanTools3D
from gpmcl.mapper import ISS3DMapper, ISS3DMapperConfig
from gpmcl.localization_scenario import (
    LocalizationScenario,
    LocalizationScenarioConfig,
    LocalizationSyncMessage,
    LocalizationPipeline,
)
from gpmcl.particle_filter import (
    ParticleFilter,
    ParticleFilterConfig,
    Pose2D,
)
from gpmcl.regression import GPRegression, GPRegressionConfig
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
        self.df_particles = pd.DataFrame(columns=["x", "y", "theta"])
        self.df_landmarks = pd.DataFrame(columns=["x", "y"])
        # a count used to print the number of iterations already performed by the filter
        self.debug_iteration_count = 0

    def initialize(self, synced_msgs: LocalizationSyncMessage) -> None:
        mapper_config = self.__get_mapper_config(self.config)
        mapper = ISS3DMapper(mapper_config)
        GP_process = self.__get_process_gp(self.config)
        pf_config = self.__get_pf_config(self.config)
        # instantiate the particle filter
        self.pf = ParticleFilter(
            config=pf_config,
            mapper=mapper,
            process_regressor=GP_process,
            initial_ground_truth=synced_msgs.groundtruth,
            initial_odom_estimate=synced_msgs.odom_est,
        )

    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        # increment the iteration counter
        self.debug_iteration_count += 1
        print(f"[{timestamp}]: iteration {self.debug_iteration_count}")
        # actual inference begins here
        pcd = ScanTools3D.scan_msg_to_open3d_pcd(synced_msgs.scan_3d)
        local_feature_map = self.pf.mapper.detect_features(pcd)
        print(f"[{timestamp}]: Detected {len(local_feature_map.features)} features.")
        # compute the prior by sampling from the GP
        self.pf.predict(U=synced_msgs.odom_est)
        # compute the posterior by incorporating map
        self.pf.update(Z=local_feature_map)
        T_updated = self.pf.mean().T
        correspondences = self.pf.mapper.correspondence_search(local_feature_map, T_updated)
        self.pf.mapper.update(local_feature_map, T_updated, correspondences)
        print(f"[{timestamp}]: Map now has {len(self.pf.mapper.get_map().features)} landmarks.")
        # append posterior to trajectory
        if synced_msgs.groundtruth:
            delta_T_error = Pose2D.from_odometry(synced_msgs.groundtruth).inv() @ T_updated
            error_norm = np.linalg.norm(Pose2D(delta_T_error).as_twist()[:2])
            print(f"[{timestamp}]: Pose error is: {error_norm}")
        # update the trajectory dataframe
        self.__update_trajectory(
            # provide current estimate as twist (x, y, theta)
            estimate=self.pf.mean().as_twist(),
            # provide ground truth pose as twist (x, y, theta) if available
            groundtruth=Pose2D.from_odometry(synced_msgs.groundtruth).as_twist()
            if synced_msgs.groundtruth is not None
            else None,
        )
        # PCD of the latest features in the updated pose frame
        if self.debug_visualize:
            feature_pcd = local_feature_map.transform(T_updated).as_pcd()
            idxs_feature_inliers = list(map(lambda c: c.idx_feature, correspondences.correspondences))
            feature_inlier_pcd = feature_pcd.select_by_index(idxs_feature_inliers)
            idxs_feature_outliers = correspondences.feature_outlier_idxs
            feature_outlier_pcd = feature_pcd.select_by_index(idxs_feature_outliers)
            ScanTools3D.visualize_scene(
                scan_pcd=pcd.transform(self.pf.mean().as_t3d()),
                map_pcd=self.pf.mapper.get_map().as_pcd(),
                feature_inlier_pcd=feature_inlier_pcd,
                feature_outlier_pcd=feature_outlier_pcd,
            )

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

    def __get_pf_config(self, config: Dict) -> ParticleFilterConfig:
        return ParticleFilterConfig.from_config(config)

    def __get_mapper_config(self, config: Dict) -> Optional[ISS3DMapperConfig]:
        return ISS3DMapperConfig.from_config(config)

    def __get_process_gp(self, config: Dict) -> GPRegression:
        # load the process GP from the config
        gp_config = GPRegressionConfig.from_config(config=config, key="process_gp")
        gp = GPRegression(gp_config)
        return gp

    def __update_trajectory(self, estimate: np.ndarray, groundtruth: Optional[np.ndarray] = None) -> None:
        """Update the dataframes containing the estimated and (optionally) ground truth trajectories."""
        # the current index is the length of the dataframe
        idx_trajectory_curr, *_ = self.df_trajectory_estimated.shape
        self.df_trajectory_estimated.loc[idx_trajectory_curr, :] = estimate
        partilces_weighted = np.hstack((self.pf.Xs, self.pf.ws.reshape(-1, 1)))
        df_particles_curr = pd.DataFrame(columns=["x", "y", "theta", "w"], data=partilces_weighted)
        self.df_particles = pd.concat((self.df_particles, df_particles_curr), ignore_index=True)
        df_landmarks_curr = pd.DataFrame(columns=["x", "y"], data=self.pf.mapper.get_map().as_matrix()[:, :2])
        self.df_landmarks = pd.concat((self.df_landmarks, df_landmarks_curr))
        # store ground truth if provided
        if groundtruth is not None:
            self.df_trajectory_groundtruth.loc[idx_trajectory_curr, :] = groundtruth


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
