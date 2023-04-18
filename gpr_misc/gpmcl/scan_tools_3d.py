from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2
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
        # points = ros_numpy.numpify(msg)
        # see: http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.__init__
        # and: >>> help(open3d.cpu.pybind.utility.Vector3dVector)
        pcd = open3d.geometry.PointCloud(
            # points=open3d.cpu.pybind.utility.Vector3dVector(points)
        )
        return pcd
