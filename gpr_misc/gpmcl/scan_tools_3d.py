from rosbags.typesys.types import (
    sensor_msgs__msg__PointCloud2 as PointCloud2,
    sensor_msgs__msg__PointField as PointField,
    std_msgs__msg__Header as Header,
)
import ros_numpy
import sensor_msgs.msg
import std_msgs.msg
import open3d

# TODO: make this work ...
# import ros_numpy


class ScanTools3D:
    """Utility class for converting and processing 3D Scan data."""

    @staticmethod
    def scan_msg_to_open3d_pcd(msg: PointCloud2) -> open3d.geometry.PointCloud:
        """convert a rosbags `PointCloud2` message to an Open3D `PointCloud` object"""
        # convert message to numpy array
        # TODO: make this work
        ros_msg = ScanTools3D.__to_rospcd(msg)
        points = ros_numpy.numpify(ros_msg)
        # see: http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.__init__
        # and: >>> help(open3d.cpu.pybind.utility.Vector3dVector)
        pcd = open3d.geometry.PointCloud(
            points=open3d.cpu.pybind.utility.Vector3dVector(points)
        )
        return pcd

    @staticmethod
    def __to_rospcd(msg: PointCloud2) -> sensor_msgs.msg.PointCloud2:
        """Convert PointCloud2 from `rosbags` to actual ROS message"""
        pcd_msg = sensor_msgs.msg.PointCloud2()
        pcd_msg.header = ScanTools3D.__to_ros_header(msg.header)
        pcd_msg.is_bigendian = msg.is_bigendian
        pcd_msg.point_step = msg.point_step
        pcd_msg.row_step = msg.row_step
        pcd_msg.data = msg.data
        # convert point fields
        point_fields = list(map(ScanTools3D.__to_ros_point_field, msg.fields))
        pcd_msg.fields = point_fields

    @staticmethod
    def __to_ros_point_field(point_field: PointField) -> sensor_msgs.msg.PointField:
        ros_pointfield = sensor_msgs.msg.PointField()
        ros_pointfield.count = point_field.count
        ros_pointfield.datatype = point_field.datatype
        ros_pointfield.name = point_field.name
        ros_pointfield.offset = point_field.offset
        return ros_pointfield

    @staticmethod
    def __to_ros_header(header: Header) -> std_msgs.msg.Header:
        ros_header = std_msgs.msg.Header()
        ros_header.frame_id = header.frame_id
        ros_header.stamp.nsecs = header.stamp.nanosec
        ros_header.stamp.secs = header.stamp.sec
        return ros_header
