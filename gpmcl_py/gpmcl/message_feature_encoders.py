"""This file contains a `(MessageType, encoder_fn)` map of all desired feature encoders"""
from gpmcl.helper_types import GPFeature, FeatureEncodeFn
from rosbags.typesys.types import (
    sensor_msgs__msg__Imu as Imu,
    geometry_msgs__msg__Twist as Twist,
    nav_msgs__msg__Odometry as Odometry,
    sensor_msgs__msg__PointCloud2 as PointCloud2,
)
from typing import List, Optional, Tuple, Any
from scipy.spatial.transform import Rotation
import numpy as np


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
    # construct numpy quaternion from pose orientation
    quat = np.array(
        [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
    )
    # convert rotation to euler angles and extract only z-rotation
    r = Rotation(quat, normalize=True)
    theta, *_ = r.as_euler("zyx", degrees=False)
    return __to_gp_features(
        [
            (f"pose2d.x ({topic})", msg.pose.pose.position.x),
            (f"pose2d.y ({topic})", msg.pose.pose.position.y),
            (f"pose2d.yaw ({topic})", theta),
            (f"twist2d.x ({topic})", msg.twist.twist.linear.x),
            (f"twist2d.y ({topic})", msg.twist.twist.linear.y),
            (f"twist2d.ang ({topic})", msg.twist.twist.angular.z),
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


def __encode_pcd(_: PointCloud2, __: str) -> List[GPFeature]:
    return __to_gp_features([("del", 0.0)])


ENCODER_MAP = {
    Imu.__msgtype__: __encode_imu,
    Twist.__msgtype__: __encode_twist,
    Odometry.__msgtype__: __encode_odometry,
    PointCloud2.__msgtype__: __encode_pcd,
}


def get_encoder(msg_type: str) -> Optional[FeatureEncodeFn]:
    return ENCODER_MAP.get(msg_type)  # type: ignore
