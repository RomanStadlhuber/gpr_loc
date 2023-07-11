from gpmcl.rosbag_sync_reader import RosbagSyncReader, SyncMessage
from gpmcl.config import BagRunnerConfig
from rosbags.typesys.types import (
    nav_msgs__msg__Odometry as Odometry,
    sensor_msgs__msg__PointCloud2 as PointCloud2,
)
from typing import Dict, Optional
from abc import ABC, abstractmethod
import pathlib


class LocalizationSyncMessage(SyncMessage):
    """A wrapper for synchronized localization in GPMCL"""

    topic_odom_groundtruth: Optional[str] = ""
    topic_odom_est: str = ""
    topic_scan_3d: str = ""

    def __init__(self, odom_est: Odometry, scan_3d: PointCloud2, groundtruth: Optional[Odometry] = None) -> None:
        self.groundtruth = groundtruth
        self.odom_est = odom_est
        self.scan_3d = scan_3d
        super().__init__()

    @staticmethod
    def set_topics(topic_odom_est: str, topic_scan_3d: str, topic_odom_groundtruth: Optional[str] = None) -> None:
        LocalizationSyncMessage.topic_odom_groundtruth = topic_odom_groundtruth
        LocalizationSyncMessage.topic_odom_est = topic_odom_est
        LocalizationSyncMessage.topic_scan_3d = topic_scan_3d

    @staticmethod
    def from_dict(d: Dict) -> Optional["LocalizationSyncMessage"]:
        # groundtruth is required but not present?
        groundtruth_not_present = (
            LocalizationSyncMessage.topic_odom_groundtruth is not None
            and LocalizationSyncMessage.topic_odom_groundtruth not in d.keys()
        )

        if (
            LocalizationSyncMessage.topic_odom_est not in d.keys()
            or LocalizationSyncMessage.topic_scan_3d not in d.keys()
            or groundtruth_not_present
        ):
            return None
        else:
            # should collect groundtruth?
            gt = LocalizationSyncMessage.topic_odom_groundtruth is not None
            # construct the message from dictionary
            return LocalizationSyncMessage(
                groundtruth=d[LocalizationSyncMessage.topic_odom_groundtruth] if gt else None,
                odom_est=d[LocalizationSyncMessage.topic_odom_est],
                scan_3d=d[LocalizationSyncMessage.topic_scan_3d],
            )


class LocalizationPipeline(ABC):
    @abstractmethod
    def initialize(self, initial_msg: LocalizationSyncMessage) -> None:
        """Initialize the Pipeline"""

    @abstractmethod
    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        """The inference step of a localization pipeline."""
        pass

    @abstractmethod
    def export_trajectory(self, out_dir: pathlib.Path) -> None:
        """Export the estimated trajectory."""
        pass


class LocalizationScenario:
    def __init__(self, config: BagRunnerConfig, pipeline: LocalizationPipeline) -> None:
        self.config = config
        self.localization_pipeline = pipeline
        # set the desired topics of the Sync Message
        LocalizationSyncMessage.set_topics(
            topic_odom_groundtruth=config["groundtruth_topic"],
            topic_odom_est=config["estimated_odometry_topic"],
            topic_scan_3d=config["pointcloud_topic"],
        )
        self.pipeline_initialized = False

    def spin_bag(self) -> None:
        """Perform localization on the bag."""
        rosbag_sync_reader = RosbagSyncReader(pathlib.Path(self.config["bag_path"]))
        # set of topic names the bag is searched for
        topic_names = set([self.config["estimated_odometry_topic"], self.config["pointcloud_topic"]])
        # add the ground truth topic if it exists
        if self.config["groundtruth_topic"] is not None:
            topic_names.add(self.config["groundtruth_topic"])
        rosbag_sync_reader.spin(
            topics=topic_names,
            callback=self.__message_callback,
            grace_period_secs=self.config["sync_period"],
            max_iterations=self.config["stop_after"] or -1,
        )

    def export_metrics(self, out_dir: pathlib.Path) -> None:
        """Export metrics of the estimation pipeline."""
        self.localization_pipeline.export_trajectory(out_dir)

    def __message_callback(self, synced_messages: Optional[Dict], timestamp: Optional[int]) -> None:
        """Convert a synced message dictionary to a localization message and run inference."""
        # skip proesssing if sync failed
        if synced_messages is None or timestamp is None:
            return

        localization_sync_message = LocalizationSyncMessage.from_dict(synced_messages)
        if localization_sync_message is None:
            return
        # run localization inference
        if not self.pipeline_initialized:
            self.localization_pipeline.initialize(localization_sync_message)
            self.pipeline_initialized = True
        else:
            self.localization_pipeline.inference(localization_sync_message, timestamp)
