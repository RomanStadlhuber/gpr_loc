from plotting.plot_trajectory_from_regression import plot_trajectory_from_regression
import pathlib

if __name__ == "__main__":
    pth_trajectory_gt = pathlib.Path("data/gpmcl/trajectory/trajectory_groundtruth.csv")
    # pth_trajectory_gt = pathlib.Path("data/trajectories/eval/taurob_eval__trajectory-poses_labels.csv")

    pth_D_eval = pathlib.Path("/home/roman/git/gpr_loc/gpr_misc/data/gpmcl/gpmcl_process_train")
    # pth_D_eval = pathlib.Path("data/regression_outputs/debug_regression_sparse")

    plot_trajectory_from_regression(pth_trajectory_gt, pth_D_eval)
