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
from typing import Optional, List, Dict
import numpy as np
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

    def __init__(self, config: Dict) -> None:
        self.config = config  # dict containing the pipeline configuration
        self.trajectory: List[np.ndarray] = []

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
        pcd = ScanTools3D.scan_msg_to_open3d_pcd(synced_msgs.scan_3d)
        # visualize the point cloud
        # ScanTools3D.visualize_pcd(pcd)
        local_feature_map = self.pf.mapper.detect_features(pcd)
        print(f"[{timestamp}]: Detected {len(local_feature_map.features)} features.")
        # compute the prior by sampling from the GP
        self.pf.predict(U=synced_msgs.odom_est)
        # compute the posterior by incorporating map
        self.pf.update(Z=local_feature_map)
        T_updated = self.pf.mean().T
        correspondences = self.pf.mapper.correspondence_search(local_feature_map, self.pf.mean().T)
        self.pf.mapper.update(local_feature_map, T_updated, correspondences)
        print(f"[{timestamp}]: Map now has {len(self.pf.mapper.get_map().features)} landmarks.")
        # append posterior to trajectory
        self.trajectory.append(T_updated)
        if synced_msgs.groundtruth:
            delta_T_error = Pose2D.from_odometry(synced_msgs.groundtruth).inv() @ T_updated
            error_norm = np.linalg.norm(Pose2D(delta_T_error).as_twist())
            print(f"[{timestamp}]: Pose error is: {error_norm}")

    def evaluate(self) -> None:
        initial_pose, *_, last_pose = self.trajectory

        x_init = Pose2D.pose_to_twist(initial_pose)
        x_last = Pose2D.pose_to_twist(last_pose)

        print(f"Initial pose (x, y, theta): {x_init}")
        print(f"Last pose (x, y, theta): {x_last}")

    def __get_pf_config(self, config: Dict) -> ParticleFilterConfig:
        return ParticleFilterConfig.from_config(config)

    def __get_mapper_config(self, config: Dict) -> Optional[ISS3DMapperConfig]:
        return ISS3DMapperConfig.from_config(config)

    def __get_process_gp(self, config: Dict) -> GPRegression:
        # load the process GP from the config
        gp_config = GPRegressionConfig.from_config(config=config, key="process_gp")
        gp = GPRegression(gp_config)
        return gp


arg_parser = argparse.ArgumentParser(
    prog="gpmcl_pipeline", description="Perform localization using gaussian process regression"
)

arg_parser.add_argument(
    "config_path",
    metavar="Path to the configuration YAML file.",
    type=str,
)

if __name__ == "__main__":
    args = arg_parser.parse_args()
    config_path = pathlib.Path(args.config_path)
    config_file = open(config_path)
    config = yaml.safe_load(config_file)
    # instantiate the pipeline
    pipeline = GPMCLPipeline(config)
    # configure the scenario
    scenario_config = LocalizationScenarioConfig.from_config(config)
    # instantiate the scenario
    localization_scenario = LocalizationScenario(config=scenario_config, pipeline=pipeline)
    # run localization inference
    localization_scenario.spin_bag()
