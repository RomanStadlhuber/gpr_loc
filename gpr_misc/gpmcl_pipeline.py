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
from typing import Optional
import numpy as np
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

    def __init__(self) -> None:
        mapper_config = self.__get_mapper_config()
        mapper = ISS3DMapper(mapper_config)
        GP_process = self.__get_process_gp()
        pf_config = self.__get_pf_config()
        # instantiate the particle filter
        self.pf = ParticleFilter(config=pf_config, mapper=mapper, process_regressor=GP_process)

    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        pcd = ScanTools3D.scan_msg_to_open3d_pcd(synced_msgs.scan_3d)
        print(f"[{timestamp}]: Got PCD with {np.shape(np.asarray(pcd.points))} points.")
        # visualize the point cloud
        # ScanTools3D.visualize_pcd(pcd)
        local_feature_map = self.pf.mapper.detect_features(pcd)
        print(f"[{timestamp}]: Detected {len(local_feature_map.features)} features.")
        # TODO: predict
        self.pf.predict(U=synced_msgs.odom_est)
        self.pf.update(Z=local_feature_map)
        # TODO: update
        updated_pose = self.pf.posterior_pose
        correspondences = self.pf.mapper.correspondence_search(local_feature_map, self.pf.posterior_pose.T)
        self.pf.mapper.update(local_feature_map, updated_pose.T, correspondences)
        print(f"[{timestamp}]: Map now has {len(self.pf.mapper.get_map().features)} landmarks.")

        # detect features

    def __get_pf_config(self) -> ParticleFilterConfig:
        # TODO: load from YAML
        R = 0.5 * np.eye(3, dtype=np.float64)
        Q = 0.2 * np.eye(3, dtype=np.float64)
        M = 50
        T0 = Pose2D.from_twist(np.zeros(3))
        return ParticleFilterConfig(
            particle_count=M,
            process_covariance_R=R,
            observation_covariance_Q=Q,
            initial_guess_pose=T0,
        )

    def __get_mapper_config(self) -> Optional[ISS3DMapperConfig]:
        # TODO: load from YAML
        return None

    def __get_process_gp(self) -> GPRegression:
        # TODO: fill-in, later load from YAML
        gp_config = GPRegressionConfig(
            model_dir=pathlib.Path(""),
            train_data_path=pathlib.Path(""),
            labels_dX_last=["", "", ""],
            labels_dU=["", "", ""],
            is_sparse=False,
        )
        gp = GPRegression(gp_config)
        return gp


if __name__ == "__main__":
    # instantiate the pipeline
    pipeline = GPMCLPipeline()
    # configure the scenario
    scenario_config = LocalizationScenarioConfig(
        bag_path=pathlib.Path("./bags/explore.bag"),  # TODO: make this configurable
        topic_odom_est="/odom",
        topic_scan_3d="/velodyne_points",
        localization_pipeline=pipeline,
    )
    # instantiate the scenario
    localization_scenario = LocalizationScenario(config=scenario_config)
    # run localization inference
    localization_scenario.spin_bag()
