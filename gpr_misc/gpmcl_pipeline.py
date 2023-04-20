from gpmcl.scan_tools_3d import ScanTools3D
from gpmcl.localization_scenario import (
    LocalizationScenario,
    LocalizationScenarioConfig,
    LocalizationSyncMessage,
    LocalizationPipeline,
)
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
        super().__init__()

    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        pcd = ScanTools3D.scan_msg_to_open3d_pcd(synced_msgs.scan_3d)
        # visualize the point cloud
        ScanTools3D.visualize_pcd(pcd)
        print(f"[{timestamp}]: Got PCD with {np.shape(np.asarray(pcd.points))} points!")


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
