from plotting.plot_trajectory_from_regression import plot_trajectory_from_regression, compute_trajectory_from_deltas
from plotting.plotters import TrajectoryPlotter, MultiHistogramPlotter
from gpmcl.helper_types import GPDataset
from typing import Optional
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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

        data_dir = pathlib.Path("data/eval_trajectories")

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
        df_landmarks = pd.read_csv(pth_landmarks) if pth_landmarks.exists() else pd.DataFrame(columns=["x", "y"])
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

    def paper_2__effective_weight_histogram(
        self,
        df_w_fastslam1: pd.DataFrame,
        df_w_fastslam2: pd.DataFrame,
        num_bins: int = 20,
        title: Optional[str] = "Effective weight comparison between FastSLAM variants.",
        w_eff_colname: str = "w_eff",
    ) -> None:
        # prepare the datasets for plotting
        ws_fastslam1 = df_w_fastslam1[w_eff_colname]
        ws_fastslam2 = df_w_fastslam2[w_eff_colname]
        hist_fastslam1, bins_fastslam1 = np.histogram(ws_fastslam1, num_bins)
        hist_fastslam2, bins_fastslam2 = np.histogram(ws_fastslam2, num_bins)
        counts = np.hstack((hist_fastslam1, hist_fastslam2))
        # stack the bin values but omit the last bin edge (or should it be the first?)
        bins = np.hstack((bins_fastslam1[:-1], bins_fastslam2[:-1]))
        variants = np.hstack(
            (
                np.repeat("FastSLAM", hist_fastslam1.shape),
                np.repeat("FastSLAM2.0", hist_fastslam2.shape),
            )
        ).astype(np.string_)
        df_hist = pd.DataFrame(
            columns=["value", "occurrences", "variant"],
            # amalgamate and transpose to create table shape
            data=np.vstack((bins, counts, variants)).T,
        )
        plotter = MultiHistogramPlotter()
        plotter.plot_data(df_hist, x="value", y="occurrences", color="variant")


if __name__ == "__main__":
    plotter = PaperFigurePlotter()
    # plotter.paper_2__compare_gp_with_ros()
    # plotter.paper_2__trajectories_from_GPs()
    data_dir = pathlib.Path("./data/eval_trajectories")
    # plotter.paper_2__compare_trajectories()
    plotter.paper_2__effective_weight_histogram(
        df_w_fastslam1=pd.DataFrame(columns=["w_eff"]),
        df_w_fastslam2=pd.read_csv(data_dir / "effective_weights.csv"),
        num_bins=50,
    )
