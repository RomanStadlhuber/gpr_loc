from plotting.plotters import TrajectoryPlotter
from helper_types import GPDataset
from transform import Pose2D
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pathlib


def plot_trajectory_from_regression(
    pth_trajectory_groundtruth: pathlib.Path,
    pth_D_eval: pathlib.Path,
) -> None:
    # load the ground truth trajectory
    df_trajectory_groundtruth = pd.read_csv(pth_trajectory_groundtruth, index_col=0)
    # load the evaluated dataset (which attempts to regress the ground truth trajectory)
    D_eval = GPDataset.load(pth_D_eval)
    # obtain the motion chanegs from the regression output
    # is a M x 3  matrix with columns x, y, theta
    motion_deltas = D_eval.labels.to_numpy()
    # the starting point of the regression trajectory
    T0 = Pose2D.from_twist(df_trajectory_groundtruth.loc[0, :].to_numpy())
    df_trajectory_estimated = __compute_trajectory_from_deltas(motion_deltas, T0)
    # plot both trajectories
    fig = go.Figure()
    plotter = TrajectoryPlotter(fontsize=10)
    gt_label_x, gt_label_y, *_ = df_trajectory_groundtruth.columns.tolist()
    color_gt = "darkgreen"
    color_est = "orange"
    lineplot_gt = plotter.line_trace(
        x=df_trajectory_groundtruth[gt_label_x].to_numpy(),
        y=df_trajectory_groundtruth[gt_label_y].to_numpy(),
        color=color_gt,
        name="Groundtruth trajectory",
    )
    lineplot_est = plotter.line_trace(
        x=df_trajectory_estimated["x"].to_numpy(),
        y=df_trajectory_estimated["y"].to_numpy(),
        color=color_est,
        name="Estimated trajectory",
    )
    fig.add_trace(lineplot_gt)
    fig.add_trace(lineplot_est)
    plotter.format_figure(
        fig,
        x_title="x [m]",
        y_title="y [m]",
        width_px=1200,
        height_px=900,
    )
    fig.show()


def __compute_trajectory_from_deltas(
    motion_deltas: np.ndarray,  # motion deltas in local frame
    T0: Pose2D = Pose2D(),  # initial guess pose
) -> pd.DataFrame:
    """Generate a trajectory with columns from motion deltas.

    Returns a `pd.Dataframe` with columns `"x"`, `"y"` and `"theta"`.
    """
    rows, *_ = motion_deltas.shape
    T_curr = T0
    # initialize the empty trajectory dataframe
    df_trajectory = pd.DataFrame(columns=["x", "y", "theta"])
    # recursively apply perturbations to the state
    for i in range(rows):
        u_i = motion_deltas[i, :]
        T_curr.perturb(u_i)
        # convert the pose to a minimal representation and store in dataframe
        df_trajectory.loc[i, :] = T_curr.as_twist()

    return df_trajectory
