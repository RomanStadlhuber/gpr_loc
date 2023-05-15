from plotting.plotters import TrajectoryPlotter
from helper_types import GPDataset
import plotly.graph_objects as go
import numpy as np
import pathlib

# from helper_types import GPDataset
# import numpy as np


def plot_trajectories() -> None:
    fig = go.Figure()
    plotter = TrajectoryPlotter(fontsize=18)

    # --- load trajectories from GPDatasets
    # region

    # NOTE: assume the column names are the same for all datasets
    path_cw = pathlib.Path("./data/trajectories/clockwise")
    path_ccw = pathlib.Path("./data/trajectories/counterclockwise")
    path_eval = pathlib.Path("./data/trajectories/eval")

    dataset_cw = GPDataset.load(dataset_folder=path_cw, name="clockwise trajectory")
    # the diff drive estimates are the features
    trajectory_cw_odom = dataset_cw.features
    # the ground truth are the labels
    trajectory_cw_groundtruth = dataset_cw.labels
    color_cw = "blue"

    dataset_ccw = GPDataset.load(dataset_folder=path_ccw, name="counter-clockwise trajectory")
    # the diff drive estimates are the features
    trajectory_ccw_odom = dataset_ccw.features
    # the ground truth are the labels
    trajectory_ccw_groundtruth = dataset_ccw.labels
    color_ccw = "green"

    dataset_eval = GPDataset.load(dataset_folder=path_eval, name="clockwise trajectory")
    # the diff drive estimates are the features
    trajectory_eval_odom = dataset_eval.features
    # the ground truth are the labels
    trajectory_eval_groundtruth = dataset_eval.labels
    color_eval = "red"
    # endregion

    # --- plot trajectiores for paper no. 1
    # region

    lineplot_cw = plotter.line_trace(
        x=trajectory_cw_groundtruth["pose2d.x (/ground_truth/odom)"].to_numpy(),
        y=trajectory_cw_groundtruth["pose2d.y (/ground_truth/odom)"].to_numpy(),
        color=color_cw,
        name="Training trajectory (clockwise)",
    )
    lineplot_ccw = plotter.line_trace(
        x=trajectory_ccw_groundtruth["pose2d.x (/ground_truth/odom)"].to_numpy(),
        y=trajectory_ccw_groundtruth["pose2d.y (/ground_truth/odom)"].to_numpy(),
        color=color_ccw,
        name="Training trajectory (counter-clockwise)",
    )
    lineplot_eval = plotter.line_trace(
        x=trajectory_eval_groundtruth["pose2d.x (/ground_truth/odom)"].to_numpy(),
        y=trajectory_eval_groundtruth["pose2d.y (/ground_truth/odom)"].to_numpy(),
        color=color_eval,
        name="Evaluation trajectory",
    )
    training_setpoints = np.array([[1.5, 2], [0, 4], [-1.5, 2], [0, 0]])
    test_setpoints = np.array([[1.5, 0], [-1.5, 4], [1.5, 4], [-1.5, 0]]) + np.array([0.5, 0])
    markers_training = plotter.marker_trace(
        y=training_setpoints[:, 0],
        x=training_setpoints[:, 1],
        symbol="triangle-down",
        color="orange",
        marker_size=16,
        marker_outline_width=2,
        name="Training setpoints (both trajectories)",
    )
    markers_test = plotter.marker_trace(
        y=test_setpoints[:, 0],
        x=test_setpoints[:, 1],
        symbol="diamond",
        color="red",
        marker_size=12,
        marker_outline_width=2,
        name="Evaluation setpoints",
    )
    fig.add_trace(lineplot_cw)
    fig.add_trace(lineplot_ccw)
    fig.add_trace(lineplot_eval)
    fig.add_trace(markers_training)
    fig.add_trace(markers_test)
    plotter.format_figure(
        fig,
        x_title="x [m]",
        y_title="y [m]",
        y_range=(-2, 4.8),
        # x_range=(-1.5, 5.5),
        width_px=800,
        height_px=600,
    )

    # endregion

    fig.show()
