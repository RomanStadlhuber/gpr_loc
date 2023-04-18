from gpmcl.rosbag_sync_reader import RosbagSyncReader, SyncMessage
from rosbags.typesys.types import (
    nav_msgs__msg__Odometry as Odometry,
    sensor_msgs__msg__PointCloud2 as PointCloud2,
)
from typing import Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pathlib


class LocalizationSyncMessage(SyncMessage):
    """A wrapper for synchronized localization in GPMCL"""

    topic_odom_est: str = ""
    topic_scan_3d: str = ""

    def __init__(self, odom_est: Odometry, scan_3d: PointCloud2) -> None:
        self.odom_est = odom_est
        self.scan_3d = scan_3d
        super().__init__()

    @staticmethod
    def set_topics(topic_odom_est: str, topic_scan_3d: str) -> None:
        LocalizationSyncMessage.topic_odom_est = topic_odom_est
        LocalizationSyncMessage.topic_scan_3d = topic_scan_3d

    @staticmethod
    def from_dict(d: Dict) -> Optional["LocalizationSyncMessage"]:
        if (
            LocalizationSyncMessage.topic_odom_est not in d.keys()
            or LocalizationSyncMessage.topic_scan_3d not in d.keys()
        ):
            return None
        else:
            return LocalizationSyncMessage(
                odom_est=d[LocalizationSyncMessage.topic_odom_est],
                scan_3d=d[LocalizationSyncMessage.topic_scan_3d],
            )


class LocalizationPipeline(ABC):
    @abstractmethod
    def inference(self, synced_msgs: LocalizationSyncMessage, timestamp: int) -> None:
        """The inference step of a localization pipeline."""
        pass


@dataclass
class LocalizationScenarioConfig:
    """Configuration wrapper for localization scenario"""

    bag_path: pathlib.Path
    topic_odom_est: str
    topic_scan_3d: str
    localization_pipeline: LocalizationPipeline


class LocalizationScenario:
    def __init__(self, config: LocalizationScenarioConfig) -> None:
        self.config = config
        # set the desired topics of the Sync Message
        LocalizationSyncMessage.set_topics(
            topic_odom_est=config.topic_odom_est,
            topic_scan_3d=config.topic_scan_3d,
        )

    def spin_bag(self) -> None:
        """Perform localization on the bag."""
        rosbag_sync_reader = RosbagSyncReader(self.config.bag_path)
        rosbag_sync_reader.spin(
            topics=set([self.config.topic_odom_est, self.config.topic_scan_3d]),
            callback=self.__message_callback,
        )

    def __message_callback(
        self, synced_messages: Optional[Dict], timestamp: Optional[int]
    ) -> None:
        """Convert a synced message dictionary to a localization message and run inference."""
        # skip proesssing if sync failed
        if synced_messages is None or timestamp is None:
            return

        localization_sync_message = LocalizationSyncMessage.from_dict(synced_messages)
        if localization_sync_message is None:
            return
        # run localization inference
        self.config.localization_pipeline.inference(
            localization_sync_message, timestamp
        )
