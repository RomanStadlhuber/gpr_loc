"""This file contains a `(MessageType, encoder_fn)` map of all desired feature encoders"""
from helper_types import GPFeature, FeatureEncodeFn
from rosbags.typesys.types import (
    sensor_msgs__msg__Imu as Imu,
    geometry_msgs__msg__Twist as Twist,
    nav_msgs__msg__Odometry as Odometry,
)
from typing import List, Optional


def __encode_imu(_: Imu, topic: str) -> List[GPFeature]:
    return [GPFeature(f"imu ({topic})", 123)]


def __encode_odometry(_: Odometry, topic: str) -> List[GPFeature]:
    return [GPFeature(f"odometry ({topic})", 456)]


def __encode_twist(_: Twist, topic: str) -> List[GPFeature]:
    return [GPFeature(f"twist ({topic})", 789)]


ENCODER_MAP = {
    Imu.__msgtype__: __encode_imu,
    Twist.__msgtype__: __encode_twist,
    Odometry.__msgtype__: __encode_odometry,
}


def get_encoder(msg_type: str) -> Optional[FeatureEncodeFn]:
    return ENCODER_MAP.get(msg_type)
