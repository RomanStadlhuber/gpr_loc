from argparse import ArgumentParser
from gpmcl.mapper import Mapper
from gpmcl.rosbag_sync_reader import RosbagSyncReader, SyncMessage
from gpmcl.config import GPMappingOfflineConfig, load_gpmapping_offline_config
from gpmcl.scan_tools_3d import ScanTools3D, PointCloudVisualizer
from gpmcl.transform import Pose2D
from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2
from typing import Dict, Optional
import pandas as pd
import numpy as np
import pathlib
import sys

arg_parser = ArgumentParser(prog="map3d_post", description="Generate a map from 3D scans in post, given a trajectory")

# the trajectory used to obtain poses
arg_parser.add_argument("-tr", "--trajectory", dest="trajectory", metavar="<path/to/trajectory.csv>", required=True)
# the bag used to load 3d scan data
arg_parser.add_argument("-conf", "--config_file", dest="config", metavar="/path/to/config.yaml", required=True)


class ScanMapperSyncMessage(SyncMessage):
    """Implements the `SyncMessage` class to read pointcloud data from rosbags."""

    topic_scan_3d: str = ""

    def __init__(self, scan_3d: PointCloud2) -> None:
        """Create a message from pointcloud data."""
        self.scan_3d = scan_3d
        super().__init__()

    @staticmethod
    def set_topic_name(topic_scan_3d: str) -> None:
        ScanMapperSyncMessage.topic_scan_3d = topic_scan_3d

    @staticmethod
    def from_dict(d: Dict) -> "ScanMapperSyncMessage":
        return ScanMapperSyncMessage(d[ScanMapperSyncMessage.topic_scan_3d])


class OffineScanMapper:
    """Generate a 3D scan map from pre-computed trajectory data.

    Uses a dataframe of the precomputed trajectory as well as the config used during SLAM.
    """

    def __init__(self, df_trajectory: pd.DataFrame, config: GPMappingOfflineConfig) -> None:
        # store the inputs
        self.config = config
        self.bag_path = pathlib.Path(self.config["bag_runner"]["bag_path"])
        self.df_trajectory = df_trajectory
        # index of current trajectory pose
        self.idx_trajectory = 0
        # number of max rows to process
        self.idx_max, *_ = self.df_trajectory.shape
        # prepare the reader
        self.topic_scan_3d = self.config["bag_runner"]["pointcloud_topic"]
        ScanMapperSyncMessage.set_topic_name(self.topic_scan_3d)
        # reader to obtain 3d scan messages
        self.bag_reader = RosbagSyncReader(self.bag_path)
        # mapper to process and align scans
        self.mapper = Mapper(self.config["mapper"])
        # visualizer to draw the map
        self.visualizer = PointCloudVisualizer()

    def callback_scan(self, msgs_dict: Optional[dict], timestamp: Optional[int]) -> None:
        """Callback that generates the 3D pointcloud and generates a map given the trajectory."""
        if msgs_dict is not None and timestamp is not None:
            # obtain 3D pointcloud
            msg_scan = ScanMapperSyncMessage.from_dict(msgs_dict)
            pcd_scan_raw = ScanTools3D.pointcloud2_to_open3d_pointcloud(msg_scan.scan_3d)
            # obtain trajectory pose
            twist_pose = np.array(self.df_trajectory.loc[self.idx_trajectory, ["x", "y", "theta"]])
            print(twist_pose)
            tf_pose = Pose2D.from_twist(twist_pose).as_t3d()
            self.idx_trajectory += 1
            # have the mapper process the scan (do not compute keypoints to speed up processing)
            self.mapper.process_scan(pcd_scan_raw)
            # then update the map given the trajectory pose
            self.mapper.update_map(pose=tf_pose)
            # draw the map
            self.visualizer.update(pcds=[self.mapper.pcd_map])
            # if there are no more poses in the trajectory, exit
            if self.idx_trajectory >= self.idx_max:
                print("Done.")
                sys.exit(0)

    def generate_scan_map(self) -> None:
        self.bag_reader.spin(topics={self.topic_scan_3d}, callback=self.callback_scan, grace_period_secs=0.1)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    config_path = pathlib.Path(args.config)
    trajectory_path = pathlib.Path(args.trajectory)
    if not config_path.exists():
        raise FileNotFoundError(f"Failed to find configuration file '{str(config_path)}'.")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Failed to find trajectory file '{str(trajectory_path)}'")
    offline_mapper = OffineScanMapper(
        # load pre-computed trajectory
        df_trajectory=pd.read_csv(trajectory_path),
        # load configuration that was used during offline SLAM
        config=load_gpmapping_offline_config(config_path),
    )
    offline_mapper.generate_scan_map()
