from plotting.plot_trajectory_from_regression import plot_trajectory_from_regression, compute_trajectory_from_deltas
from plotting.plotters import TrajectoryPlotter
from helper_types import GPDataset
import plotly.graph_objects as go
import pandas as pd
import pathlib


class PaperFigurePlotter:
    def __init__(self) -> None:
        pass

    def paper_2__compare_gp_with_ros(self) -> None:
        """Compare the ground truth trajectory from the ROS messages to that of the gaussian process.

        If there is an alignment offser, the frames are computed incorrectly.
        """

        df_trajectory_gt = pd.read_csv(pathlib.Path("data/gpmcl/trajectory/trajectory_groundtruth.csv"), index_col=0)
        rows, *_ = df_trajectory_gt.shape

        D_eval = GPDataset.load(pathlib.Path("data/gpmcl/gpmcl_process_eval_on_train"))
        # trimmed version to make it visually more matching
        D_eval_trimmed = GPDataset("sparse gp eval", D_eval.features.iloc[:rows], D_eval.labels.iloc[:rows])

        plot_trajectory_from_regression(
            df_trajectory_gt,
            D_eval_trimmed,
        )

    def paper_2__trajectories_from_GPs(self) -> None:
        """Plot the trajectories from the gaussian process test input and eval output data.

        If there is no alignment issue, there is a problem with the frames
        """
        D_test = GPDataset.load(pathlib.Path("data/gpmcl/datasets/gpmcl_process_test_2"))
        df_trajectory_gt = compute_trajectory_from_deltas(D_test.labels.to_numpy())
        D_eval = GPDataset.load(pathlib.Path("data/gpmcl/datasets/gpmcl_process_eval_2"))

        plot_trajectory_from_regression(
            df_trajectory_gt,
            D_eval,
        )

    def paper_2__compare_trajectories(self) -> None:
        # -- load preprocessed trajectiores from GPMCL dataframes (paper no. 2)
        # region

        data_dir = pathlib.Path("data/gpmcl/trajectories/trajectory_gt_update")

        pth_trajectory_est = pathlib.Path(data_dir / "trajectory_estimated.csv")
        pth_trajectory_groundtruth = pathlib.Path(data_dir / "trajectory_groundtruth.csv")
        pth_trajectory_odometry = pathlib.Path(data_dir / "trajectory_odometry.csv")
        pth_particles = pathlib.Path(data_dir / "particles.csv")
        pth_landmarks = pathlib.Path(data_dir / "landmarks.csv")

        fig = go.Figure()
        plotter = TrajectoryPlotter(fontsize=18)
        # endregion
        # --- plot trajectories for GPMCL paper
        # region
        trajectory_estimated = pd.read_csv(pth_trajectory_est)
        trajectory_groundtruth = pd.read_csv(pth_trajectory_groundtruth)
        trajectory_odometry = pd.read_csv(pth_trajectory_odometry)
        df_particles = pd.read_csv(pth_particles)
        df_landmarks = pd.read_csv(pth_landmarks)
        color_est = "orange"
        color_gt = "darkgreen"
        color_odom = "purple"
        lineplot_est = plotter.line_trace(
            x=trajectory_estimated["x"].to_numpy(),
            y=trajectory_estimated["y"].to_numpy(),
            color=color_est,
            name="Estimated trajectory",
        )
        lineplot_gt = plotter.line_trace(
            x=trajectory_groundtruth["x"].to_numpy(),
            y=trajectory_groundtruth["y"].to_numpy(),
            color=color_gt,
            name="Groundtruth trajectory",
        )
        lineplot_odom = plotter.line_trace(
            x=trajectory_odometry["x"].to_numpy(),
            y=trajectory_odometry["y"].to_numpy(),
            color=color_odom,
            name="Odometry estimates",
        )
        markers_particles = plotter.marker_trace(
            x=df_particles["x"],
            y=df_particles["y"],
            symbol="circle",
            color="black",
            marker_size=6,
            marker_outline_width=1,
            name="Particles",
        )
        markers_landmarks = plotter.marker_trace(
            x=df_landmarks["x"],
            y=df_landmarks["y"],
            symbol="diamond",
            color="blue",
            marker_size=10,
            marker_outline_width=1,
            name="Landmarks",
        )

        fig.add_trace(markers_particles)
        fig.add_trace(markers_landmarks)
        fig.add_trace(lineplot_est)
        fig.add_trace(lineplot_gt)
        fig.add_trace(lineplot_odom)
        plotter.format_figure(
            fig,
            x_title="x [m]",
            y_title="y [m]",
            # y_range=(-2, 4.8),
            # x_range=(-1.5, 5.5),
            width_px=1200,
            height_px=900,
        )
        # endregion

        fig.show()


if __name__ == "__main__":
    plotter = PaperFigurePlotter()
    # plotter.paper_2__compare_gp_with_ros()
    # plotter.paper_2__trajectories_from_GPs()
    plotter.paper_2__compare_trajectories()
