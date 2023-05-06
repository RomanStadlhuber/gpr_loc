from rosbags.typesys.types import (
    sensor_msgs__msg__PointCloud2 as PointCloud2,
    sensor_msgs__msg__PointField as PointField,
    std_msgs__msg__Header as Header,
)
from typing import Optional, List
import numpy as np
import ros_numpy
import sensor_msgs.msg
import std_msgs.msg
import open3d

# TODO: make this work ...
# import ros_numpy


class ScanTools3D:
    """Utility class for converting and processing 3D Scan data."""

    @staticmethod
    def visualize_pcd(pcd: open3d.geometry.PointCloud) -> None:
        """A blocking call to Open3Ds visualization pipeline."""
        open3d.visualization.draw_geometries(
            [pcd],
            front=[-0.86620647140619078, -0.23940427344046508, 0.43860226031391952],
            lookat=[-1.9334621429443359, 5.630396842956543, 0.42972373962402344],
            up=[0.31166516566442265, 0.42724517318096378, 0.84872044072529351],
            zoom=0.49999999999999978,
        )

    @staticmethod
    def visualize_scene(
        scan_pcd: open3d.geometry.PointCloud,
        map_pcd: open3d.geometry.PointCloud,
        feature_pcd: open3d.geometry.PointCloud,
    ) -> None:
        # set colors of the output PCDs
        ScanTools3D.__set_pcd_color(scan_pcd, 0.6 * np.ones((3)))  # grey
        ScanTools3D.__set_pcd_color(map_pcd, np.array([0, 0, 1.0]))  # blue
        ScanTools3D.__set_pcd_color(feature_pcd, np.array([0, 1.0, 0]))  # green
        open3d.visualization.draw_geometries(
            [scan_pcd, map_pcd, feature_pcd],
            front=[-0.86620647140619078, -0.23940427344046508, 0.43860226031391952],
            lookat=[-1.9334621429443359, 5.630396842956543, 0.42972373962402344],
            up=[0.31166516566442265, 0.42724517318096378, 0.84872044072529351],
            zoom=0.49999999999999978,
        )

    @staticmethod
    def scan_msg_to_open3d_pcd(
        msg: PointCloud2,
        scan_tf: Optional[np.ndarray] = None,
    ) -> open3d.geometry.PointCloud:
        """convert a rosbags `PointCloud2` message to an Open3D `PointCloud` object

        Optionally, use `scan_tf` (4x4 transformation) to transform the point cloud into the desired frame.
        """
        # convert message to numpy array
        # TODO: make this work
        ros_msg = ScanTools3D.__to_rospcd(msg)
        # an intermediate representation of the point cloud with a weird type and additional information
        pcd_intermediate = ros_numpy.numpify(ros_msg)
        # convert the intermediate representation to (N x 3) points-matrix
        points = ScanTools3D.__numpyfied_to_xyz(pcd_intermediate)
        # see: http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.__init__
        # and: >>> help(open3d.cpu.pybind.utility.Vector3dVector)
        pcd = open3d.geometry.PointCloud(points=open3d.cpu.pybind.utility.Vector3dVector(points))
        if scan_tf is not None:
            pcd.transform(scan_tf)
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
        pcd_msg.width = msg.width
        pcd_msg.height = msg.height
        # convert point fields
        point_fields = list(map(ScanTools3D.__to_ros_point_field, msg.fields))
        pcd_msg.fields = point_fields
        return pcd_msg

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

    @staticmethod
    def __numpyfied_to_xyz(points: np.ndarray):
        """Convert the result of `ros_numpy.numpify` to an `(N x 3)` matrix.

        `ros_numpy.numpify(point_cloud_2)` outputs a single-dimensional array of size `N` (number of points).
        Each entry is of the shape `(x, y, z, intensity, ring, theta)` and of type `numpy.void`.
        To obtain the x y and z values from an element, it needs to be adressed as `(pt[0], pt[1], pt[2])`.

        The output of this function can be used to set the points of an `open3d.geometry.PointCloud`.
        """
        num_pts, *_ = np.shape(points)
        xyz = np.zeros((num_pts, 3), dtype=np.float64)

        for idx, pt in enumerate(points):
            point = np.array([pt[0], pt[1], pt[2]], dtype=np.float64)
            xyz[idx, :] = point

        return xyz

    @staticmethod
    def __set_pcd_color(pcd: open3d.geometry.PointCloud, color: np.ndarray) -> None:
        """Set the uniform color of points."""
        points = np.asarray(pcd.points)
        num_points, *_ = np.shape(points)
        colors = np.repeat([color], repeats=num_points, axis=0)
        pcd.colors = open3d.cpu.pybind.utility.Vector3dVector(colors)


# class PointCloudVisualizer:
#     def __init__(self) -> None:
#         self.vis = open3d.visualization.Visualizer()
#         self.started = False
#
#     def update(self, pcds: List[open3d.geometry.PointCloud]) -> None:
#         """Update the visualizer with a (consistent) set of point clouds."""
#         if not self.started:
#             self.vis.create_window()
#             self.__add_pcds(pcds)
#         else:
#             self.__update_pcds(pcds)
#         self.__update_window()
#
#     def terminate(self) -> None:
#         self.vis.destroy_window()
#
#     def __add_pcds(self, pcds: List[open3d.geometry.PointCloud]) -> None:
#         """Add an individual PCD to the visualizers window."""
#         for pcd in pcds:
#             self.vis.add_geometry(pcd)
#         self.started = True
#
#     def __update_pcds(self, pcds: List[open3d.geometry.PointCloud]) -> None:
#         """Update an already displayed PCD."""
#         for pcd in pcds:
#             self.vis.update_geometry(pcd)
#
#     def __update_window(self) -> None:
#         self.vis.poll_events()
#         self.vis.update_renderer()
