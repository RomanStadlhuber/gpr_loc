from plotting.plot_trajectory_from_regression import plot_trajectory_from_regression
from helper_types import GPDataset
import pandas as pd
import pathlib

if __name__ == "__main__":
    trajectory_gt = pd.read_csv(pathlib.Path("data/gpmcl/trajectory/trajectory_groundtruth.csv"), index_col=0)
    # pth_trajectory_gt = pathlib.Path("data/trajectories/eval/taurob_eval__trajectory-poses_labels.csv")

    D_eval = GPDataset.load(pathlib.Path("data/gpmcl/gpmcl_process_sparse_eval"))
    # trimmed version to make it visually more matching
    D_eval_trimmed = GPDataset("sparse gp eval", D_eval.features.iloc[:500], D_eval.labels.iloc[:500])
    # pth_D_eval = pathlib.Path("data/regression_outputs/debug_regression_sparse")

    plot_trajectory_from_regression(
        trajectory_gt,
        D_eval_trimmed,
    )
