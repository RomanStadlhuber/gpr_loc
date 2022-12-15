"""This file contains a `(MessageType, encoder_fn)` map of all desired feature encoders"""
from helper_types import GPFeature, FeatureEncodeFn
from rosbags.typesys.types import (
    sensor_msgs__msg__Imu as Imu,
    geometry_msgs__msg__Twist as Twist,
    nav_msgs__msg__Odometry as Odometry,
)
from typing import List, Optional, Tuple, Any


def __to_gp_features(xs: List[Tuple[str, Any]]) -> List[GPFeature]:
    """convert a list of `(name, value)` tuples to GP features"""
    return list(map(lambda tup: GPFeature(column_name=tup[0], value=tup[1]), xs))


def __encode_imu(msg: Imu, topic: str) -> List[GPFeature]:
    return __to_gp_features(
        [
            (f"imu.lin_acc.x ({topic})", msg.linear_acceleration.x),
            (f"imu.lin_acc.y ({topic})", msg.linear_acceleration.y),
            (f"imu.lin_acc.z ({topic})", msg.linear_acceleration.z),
            (f"imu_ang_vel.x ({topic})", msg.angular_velocity.x),
            (f"imu_ang_vel.y ({topic})", msg.angular_velocity.y),
            (f"imu_ang_vel.z ({topic})", msg.angular_velocity.z),
        ]
    )


def __encode_odometry(msg: Odometry, topic: str) -> List[GPFeature]:
    return __to_gp_features(
        [
            (f"odom.twist.lin.x ({topic})", msg.twist.twist.linear.x),
            (f"odom.twist.lin.y ({topic})", msg.twist.twist.linear.y),
            # (f"odom.twist.lin.z ({topic})", msg.twist.twist.linear.z),
            # (f"odom.twist.ang.x ({topic})", msg.twist.twist.angular.x),
            # (f"odom.twist.ang.y ({topic})", msg.twist.twist.angular.y),
            (f"odom.twist.ang.z ({topic})", msg.twist.twist.angular.z),
        ]
    )


def __encode_twist(msg: Twist, topic: str) -> List[GPFeature]:
    return __to_gp_features(
        [
            (f"twist.lin.x ({topic})", msg.linear.x),
            (f"twist.lin.y ({topic})", msg.linear.y),
            # (f"twist.lin.z ({topic})", msg.linear.z),
            # (f"twist.ang.x ({topic})", msg.angular.x),
            # (f"twist.ang.y ({topic})", msg.angular.y),
            (f"twist.ang.z ({topic})", msg.angular.z),
        ]
    )


ENCODER_MAP = {
    Imu.__msgtype__: __encode_imu,
    Twist.__msgtype__: __encode_twist,
    Odometry.__msgtype__: __encode_odometry,
}


def get_encoder(msg_type: str) -> Optional[FeatureEncodeFn]:
    return ENCODER_MAP.get(msg_type)
