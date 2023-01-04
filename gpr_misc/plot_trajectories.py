from plotters import TrajectoryPlotter
from helper_types import GPDataset
import plotly.graph_objects as go
import pathlib


if __name__ == "__main__":
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

    dataset_ccw = GPDataset.load(
        dataset_folder=path_ccw, name="counter-clockwise trajectory"
    )
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

    # NOTE: here I deliberately switched x and y to gain space for the legend and
    # the positions do not matter since only deltas are used

    fig = go.Figure()
    plotter = TrajectoryPlotter(fontsize=18)
    lineplot_cw = plotter.line_trace(
        y=trajectory_cw_odom["pose2d.x (/odom)"].to_numpy(),
        x=trajectory_cw_odom["pose2d.y (/odom)"].to_numpy(),
        color=color_cw,
        name="Training trajectory (clockwise and counter-clockwise)",
    )
    lineplot_ccw = plotter.line_trace(
        y=trajectory_ccw_odom["pose2d.x (/odom)"].to_numpy(),
        x=trajectory_ccw_odom["pose2d.y (/odom)"].to_numpy(),
        color=color_ccw,
        name="counter clockwise",
    )
    lineplot_eval = plotter.line_trace(
        y=trajectory_eval_odom["pose2d.x (/odom)"].to_numpy(),
        x=trajectory_eval_odom["pose2d.y (/odom)"].to_numpy(),
        color=color_eval,
        name="Evaluation trajectory",
    )
    fig.add_trace(lineplot_cw)
    # fig.add_trace(lineplot_ccw)
    fig.add_trace(lineplot_eval)
    plotter.format_figure(
        fig,
        x_title="x [m]",
        y_title="y [m]",
        y_range=(-2, 2.5),
        # x_range=(-1.5, 5.5),
        width_px=800,
        height_px=600,
    )
    fig.show()
