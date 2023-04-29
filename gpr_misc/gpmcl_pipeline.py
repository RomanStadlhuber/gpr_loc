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
from typing import Optional, List
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
        self.trajectory: List[np.ndarray] = []

    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        pcd = ScanTools3D.scan_msg_to_open3d_pcd(synced_msgs.scan_3d)
        print(f"[{timestamp}]: Got PCD with {np.shape(np.asarray(pcd.points))} points.")
        # visualize the point cloud
        # ScanTools3D.visualize_pcd(pcd)
        local_feature_map = self.pf.mapper.detect_features(pcd)
        print(f"[{timestamp}]: Detected {len(local_feature_map.features)} features.")
        # compute the prior by sampling from the GP
        self.pf.predict(U=synced_msgs.odom_est)
        # compute the posterior by incorporating map
        self.pf.update(Z=local_feature_map)
        updated_pose = self.pf.mean().T
        correspondences = self.pf.mapper.correspondence_search(local_feature_map, self.pf.mean().T)
        self.pf.mapper.update(local_feature_map, updated_pose, correspondences)
        print(f"[{timestamp}]: Map now has {len(self.pf.mapper.get_map().features)} landmarks.")
        # append posterior to trajectory
        self.trajectory.append(updated_pose)

    def evaluate(self) -> None:
        initial_pose, *_, last_pose = self.trajectory

        x_init = Pose2D.pose_to_twist(initial_pose)
        x_last = Pose2D.pose_to_twist(last_pose)

        print(f"Initial pose (x, y, theta): {x_init}")
        print(f"Last pose (x, y, theta): {x_last}")

    def __get_pf_config(self) -> ParticleFilterConfig:
        # TODO: load from YAML
        R = 0.2 * np.eye(3, dtype=np.float64)
        Q = 0.1 * np.eye(3, dtype=np.float64)
        M = 5
        T0 = Pose2D.from_twist(np.array([9.417, 9.783, 2.49978]))
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
            model_dir=pathlib.Path("data/models/dense_ARD"),
            training_data_dirs=[pathlib.Path("data/process_train_ccw"), pathlib.Path("data/process_train_cw")],
            labels_dX_last=[
                "delta2d.x (/ground_truth/odom)",
                "delta2d.y (/ground_truth/odom)",
                "delta2d.z (/ground_truth/odom)",
            ],
            labels_dU=[
                "delta2d.x (/odom)",
                "delta2d.y (/odom)",
                "delta2d.z (/odom)",
            ],
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
        topic_odom_groundtruth="/ground_truth/odom",
        localization_pipeline=pipeline,
    )
    # instantiate the scenario
    localization_scenario = LocalizationScenario(config=scenario_config)
    # run localization inference
    localization_scenario.spin_bag()
