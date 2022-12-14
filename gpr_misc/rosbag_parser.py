# from rosbags.rosbag1 import Reader
# from rosbags.serde import deserialize_cdr, ros1_to_cdr
from helper_types import TRosMsg, GPFeature, FeatureEncodeFn, GPDataset
from message_feature_encoders import EncoderMap
from typing import Generic, List, Optional
import pathlib


class Message(Generic[TRosMsg]):
    """wrapper class for a ROS message"""

    def __init__(
        self,
        msg: TRosMsg,  # the deserialized message object
        fn_encode_feature: FeatureEncodeFn,  # the feature encoding fn
    ) -> None:
        self.msg = msg
        self.fn_encode_feature = fn_encode_feature
        super().__init__()

    def encode_feature(self) -> List[GPFeature]:
        """obtain a list of (name, value) GP-features from the message"""
        return self.fn_encode_feature(self.msg)


class RosbagReader:
    """helper class used to read rosbags"""

    def __init__(self, bagfile_path: pathlib.Path, encoder_map: EncoderMap) -> None:
        self.bagfile_path = bagfile_path
        pass

    def encode_bag(
        self,
        timestamp_min: Optional[int] = None,
        timestamp_max: Optional[int] = None,
    ) -> GPDataset:
        """read the rosbag and encode all"""
        pass
