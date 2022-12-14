"""This file contains a `(MessageType, encoder_fn)` map of all desired feature encoders"""
from helper_types import GPFeature, FeatureEncodeFn
from rosbags.typesys.types import sensor_msgs__msg__Imu as Imu
from typing import List, Type, Optional


class EncoderMap:
    """a singleton class containing a map from all encodable types to their encoding functions"""

    @staticmethod
    def __encode_imu(_: Imu) -> List[GPFeature]:
        return []

    __encoder_map = {Imu: __encode_imu}

    @staticmethod
    def get_encoder(type: Type) -> Optional[FeatureEncodeFn]:
        return EncoderMap.__encoder_map.get(type)
