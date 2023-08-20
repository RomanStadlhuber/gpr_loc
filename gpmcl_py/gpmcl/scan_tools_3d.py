from rosbags.typesys.types import (
    sensor_msgs__msg__PointCloud2 as PointCloud2,
)
from typing import List, Tuple
import numpy as np
import open3d
import struct

# TODO: make this work ...
# import ros_numpy


class ScanTools3D:
    """Utility class for converting and processing 3D Scan data."""

    @staticmethod
    def pointcloud2_to_open3d_pointcloud(ros_pcd: PointCloud2) -> open3d.geometry.PointCloud:
        """Convert a `sensor_msgs/PointCloud2` to an `open3d.geometry.PointCloud`.

        ### Remark
        This method was adapted from
        [Open3D-SLAMs Consversion](https://github.com/leggedrobotics/open3d_slam/blob/07a37d012516e2baac6588220c3d421e55e26e0d/open3d_utils/open3d_conversions/src/open3d_conversions.cpp#L59).
        """
        # the raw data describing the point cloud
        data_raw = ros_pcd.data
        # fields describe how the raw data is aligned to represent individual points
        data_fields = ros_pcd.fields
        # stride is the byte offset between start of a new point unit
        stride = ros_pcd.point_step
        # load the data-offsets for the x, y and z fields
        offset_x = next(f.offset for f in data_fields if f.name == "x")
        offset_y = next(f.offset for f in data_fields if f.name == "y")
        offset_z = next(f.offset for f in data_fields if f.name == "z")
        points_xyz = np.empty((ros_pcd.width * ros_pcd.height, 3), dtype=np.float32)
        idx_pt = 0
        for i in range(0, len(data_raw), stride):
            # unpacking, where "f" means to interpet as 32-bit floating point data
            # the result of unpack_from is always a tuple, but we're just interested in the first entry
            # see: https://docs.python.org/3.8/library/struct.html#struct.unpack_from
            x, *_ = struct.unpack_from("f", data_raw, i + offset_x)  # type: ignore
            y, *_ = struct.unpack_from("f", data_raw, i + offset_y)  # type: ignore
            z, *_ = struct.unpack_from("f", data_raw, i + offset_z)  # type: ignore
            points_xyz[idx_pt] = np.array([x, y, z])
            idx_pt += 1
        pcd_o3d = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points_xyz))
        return pcd_o3d

    @staticmethod
    def set_pcd_color(pcd: open3d.geometry.PointCloud, color: np.ndarray) -> None:
        """Set the uniform color of points."""
        points = np.asarray(pcd.points)
        num_points, *_ = np.shape(points)
        colors = np.repeat([color], repeats=num_points, axis=0)
        pcd.colors = open3d.cpu.pybind.utility.Vector3dVector(colors)


class PointCloudVisualizer:
    def __init__(self) -> None:
        self.vis = open3d.visualization.Visualizer()
        self.started = False
        # this is needed once to initialize the view-point
        self.geometry_added = False

    def update(self, pcds: List[open3d.geometry.PointCloud]) -> None:
        """Update the visualizer with a (consistent) set of point clouds."""
        if not self.started:
            # create a named window that is smaller than your average screen
            # I found it to be quite annoying when it's too large
            self.vis.create_window(window_name="3D Scan Map", width=800, height=600)
            self.started = True
        else:
            self.__update_pcds(pcds)
            self.__update_window()

    def terminate(self) -> None:
        self.vis.destroy_window()

    @staticmethod
    def visualize_single(pcds: List[Tuple[open3d.geometry.PointCloud, np.ndarray],]) -> None:
        """Method to visualize a set of pointclouds in a separate window.

        Pass arguments as a list of `(pcd, RGB_color)` tuples.

        This is a blocking, 'single-shot' method for visualization. Use this for debugging
        of individual pointclouds throughout the algorithm."""
        # at first, paint all the PCDs
        for pcd, color in pcds:
            pcd.paint_uniform_color(color)
        # remove the colors from the input arg
        pcds = list(map(lambda tup: tup[0], pcds))
        # visualize them
        open3d.visualization.draw_geometries(
            pcds,
            # NOTE: for some reason, these args don't work...
            # http://www.open3d.org/docs/latest/python_api/open3d.visualization.draw_geometries.html
            # zoom=float(1.0),
            # up=np.array([0, 0, 1], dtype=np.float64),
            # front=np.array([1, 0, 0], dtype=np.float64),
            width=int(800),
            height=int(600),
        )

    def __update_pcds(self, pcds: List[open3d.geometry.PointCloud]) -> None:
        """Update an already displayed PCD.

        This removes all geometry and adds it anew.
        Unfortunately, this is the only way that it will work right now.
        """
        self.vis.clear_geometries()
        for pcd in pcds:
            self.vis.add_geometry(pcd, reset_bounding_box=(not self.geometry_added))
            self.vis.get_view_control().set_up([0, 0, 1])
        if not self.geometry_added:
            self.geometry_added = True

    def __update_window(self) -> None:
        self.vis.poll_events()
        self.vis.update_renderer()
