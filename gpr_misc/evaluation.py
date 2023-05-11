from plotting.plotters import TrajectoryPlotter
from helper_types import GPDataset
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error
from typing import Optional, List
import plotly.graph_objects as go
import numpy as np
import argparse
import pathlib


class DenseToSparseEvaluation:
    def __init__(self, plotter=TrajectoryPlotter) -> None:
        self.plotter = plotter

    def compute_rmse(
        self,
        dir_dense: pathlib.Path,
        dir_sparse: pathlib.Path,
        dir_groundtruth: pathlib.Path,
    ) -> None:
        D_dense = GPDataset.load(dataset_folder=dir_dense, name="dense regression")
        D_sparse = GPDataset.load(dataset_folder=dir_sparse, name="sparse regression")
        D_groundtruth = GPDataset.load(dir_groundtruth, name="groundtruth")
        labels = D_groundtruth.labels.columns.to_list()
        print("--- root mean squared error from GP mean to groundtruth:")
        for label in labels:
            y_true = D_groundtruth.get_Y(label)
            rmse_dense = mean_squared_error(y_true=y_true, y_pred=D_dense.get_Y(label), squared=False)
            rmse_sparse = mean_squared_error(y_true=y_true, y_pred=D_sparse.get_Y(label), squared=False)
            print(f"| {label} | dense: {rmse_dense} | sparse: {rmse_sparse} |")

    def fit_linear_model_dense_sparse(
        self,
        dir_dense: pathlib.Path,
        dir_sparse: pathlib.Path,
        messages: bool = True,
        plot_titles: Optional[List[str]] = None,
    ) -> None:
        D_dense = GPDataset.load(dataset_folder=dir_dense, name="eval-dense")
        D_sparse = GPDataset.load(dataset_folder=dir_sparse, name="eval-sparse")

        label_names = D_dense.labels.columns.tolist()

        if label_names != D_sparse.labels.columns.tolist():
            raise Exception(
                """Unable to compare GP datasets!
Outputs (labels) do not have the same column names!
            """
            )

        if messages:
            print(
                """
---
Starting evaluation, remark:
    Comparing the similarity between dense and sparse outputs.
    Assuming a linear map "dense := f(sparse) = k * sparse + d".
    The zero-hypothesis is that x and y are equal (H0: k = 1).
    Furthermore, the intercept is required to be zero (d = 0).
---
            """
            )

        for idx, name in enumerate(label_names):
            # "design matrix" of the form
            # [ [1, x1],
            #   [1, x2],
            #   [1, ...],
            #   [1, xn]
            # ]
            # where the first value is the weight (should be 1)
            # the inputs are the sparse regressions
            x = D_dense.get_Y(name).reshape(-1)  # collapse to 1-dim (row-vec)
            M = x[:, np.newaxis] ** [0, 1]
            # output vector for each dense regression index
            y = D_sparse.get_Y(name).reshape(-1)  # collapse to 1-dim (row-vec)
            # perform linear least squares regression on the (sparse, dense) data
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
            (slope, _), *_ = lstsq(M, y)  # TODO: it seems that regression outputs are NaN!
            if messages:
                print(
                    f"""
---
Linear Model Regression for label "{name}":
    regression points: {x.shape[0]}
    linear model slope: {slope}
---
                """
                )

            fig = go.Figure()
            regression_points = self.plotter.marker_trace(
                x=x,
                y=y,
                symbol="cross",
                color="blue",
                name="GP outputs",
                marker_size=5,
                marker_outline_width=1,
            )
            x_min = x.min()
            x_max = x.max()  # plot the slope line from 0 to max
            # NOTE: y_max = slope * x_max
            regression_slope = self.plotter.line_trace(
                x=np.array([x_min, x_max]),
                y=np.array([(1 - slope) * x_min, (1 - slope) * x_max]),
                color="red",
                name="linear model",
            )
            fig.add_trace(regression_points)
            fig.add_trace(regression_slope)

            self.plotter.format_figure(
                fig=fig,
                figure_title=plot_titles[idx] if plot_titles else None,
                x_title="Dense GP output",
                y_title="Sparse GP output",
                width_px=800,
                height_px=800,
            )
            fig.show()


arg_parser = argparse.ArgumentParser(
    prog="evaluation",
    description="evaluate GPR outputs by comparing dense and sparse results",
)

# first positional argument is the dense data
arg_parser.add_argument(
    dest="dense_dir",
    type=str,
    metavar="Dense Regression Dataset Directory",
    help="The root directory of the dense output data",
)
# second positional argument is the sparse data
arg_parser.add_argument(
    dest="sparse_dir",
    type=str,
    metavar="Sparse Regression Dataset Directory",
    help="The root directory of the sparse output data",
)
# ground truth dataset
arg_parser.add_argument(
    "-gt",
    "--groundtruth",
    type=str,
    dest="groundtruth_dir",
    required=False,
    metavar="Groundtruth dataset",
    help="Root directory of the ground truth dataset",
)
arg_parser.add_argument(
    "-lm",
    "--linear_model_fit",
    dest="fit_linear_model",
    action="store_true",
    default=False,
    required=False,
    help="Compute and plot a linear model fit.",
)
arg_parser.add_argument(
    "-rmse",
    "--compute_rmse",
    dest="compute_rmse",
    action="store_true",
    default=False,
    required=False,
    help="Compute and print the root mean squared error (RMSE)",
)

if __name__ == "__main__":
    args = arg_parser.parse_args()
    dense_dir = pathlib.Path(args.dense_dir)
    sparse_dir = pathlib.Path(args.sparse_dir)
    groundtruth_dir = pathlib.Path(args.groundtruth_dir) or None
    fit_linear_model = args.fit_linear_model or False
    compute_rmse = args.compute_rmse or False
    # create a plotter for the output and pass to the eval processor
    plotter = TrajectoryPlotter(fontsize=20)
    evaluator = DenseToSparseEvaluation(plotter=plotter)
    if fit_linear_model:
        evaluator.fit_linear_model_dense_sparse(dir_dense=dense_dir, dir_sparse=sparse_dir)
    if compute_rmse and groundtruth_dir is not None:
        evaluator.compute_rmse(dir_dense=dense_dir, dir_sparse=sparse_dir, dir_groundtruth=groundtruth_dir)
