from gpmcl.transform import Pose2D
from gpmcl.scan_tools_3d import PointCloudVisualizer
import pandas as pd
import open3d
import pathlib


PCD_DIR = pathlib.Path("data/pcds")
TRAJ_DIR = pathlib.Path("data/eval_trajectories/trajectory_estimated.csv")

if __name__ == "__main__":
    df_trajectory = pd.read_csv(TRAJ_DIR)
    waypoints = df_trajectory[["x", "y", "theta"]].to_numpy()
    tfs = list(map(lambda twist: Pose2D.from_twist(twist).as_t3d(), waypoints))

    pcd_map = open3d.geometry.PointCloud()
    viz = PointCloudVisualizer()
    # join trajectory and pcds
    for idx_waypoint in range(waypoints.shape[0]):
        file_pcd = PCD_DIR / f"{idx_waypoint}.xyz"
        pcd_curr = open3d.io.read_point_cloud(str(file_pcd))
        tf_curr = tfs[idx_waypoint]
        pcd_curr.transform(tf_curr)
        pcd_map += pcd_curr
        # downsample the merged PCDs to remove duplicate points
        # (see: http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html#Make-a-combined-point-cloud)
        pcd_map = pcd_map.voxel_down_sample(voxel_size=0.05)
        # further reduce the map point size by removing duplicated and invalid points
        pcd_map.remove_duplicated_points()
        pcd_map.remove_non_finite_points()
        viz.update(pcds=[pcd_map], width=1600, height=1000)
