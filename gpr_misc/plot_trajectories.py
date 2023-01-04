from helper_types import GPDataset
import plotly.graph_objects as go
import pathlib


if __name__ == "__main__":
    # NOTE: assume the column names are the same for all datasets
    path_cw = pathlib.Path("./data/trajectories/clockwise")

    dataset_cw = GPDataset.load(dataset_folder=path_cw, name="Clockwise trajectory")
    # the diff drive estimates are the features
    trajectory_cw_odom = dataset_cw.features
    # the ground truth are the labels
    trajectory_cw_groundtruth = dataset_cw.labels

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trajectory_cw_odom["pose2d.x (/odom)"].to_numpy(),
            y=trajectory_cw_odom["pose2d.y (/odom)"].to_numpy(),
            line=dict(color="blue"),
            showlegend=False,
        )
    )
    print("plotting")
    fig.show()
