from plotters import TrajectoryPlotter
from helper_types import GPDataset
from scipy.linalg import lstsq
from typing import Optional, List
import plotly.graph_objects as go
import numpy as np
import argparse
import pathlib


class DenseToSparseEvaluation:
    def __init__(self, plotter=TrajectoryPlotter) -> None:
        self.plotter = plotter

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
            x = D_sparse.get_Y(name).reshape(-1)  # collapse to 1-dim (row-vec)
            M = x[:, np.newaxis] ** [0, 1]
            # output vector for each dense regression index
            y = D_sparse.get_Y(name).reshape(-1)  # collapse to 1-dim (row-vec)
            # perform linear least squares regression on the (sparse, dense) data
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
            (slope, _), *_ = lstsq(
                M, y
            )  # TODO: it seems that regression outputs are NaN!
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
    metavar="The Sparse Regression Dataset Directory",
    help="The root directory of the sparse output data",
)
# TODO: optional arguments for plotting


if __name__ == "__main__":
    args = arg_parser.parse_args()
    dense_dir = pathlib.Path(args.dense_dir)
    sparse_dir = pathlib.Path(args.sparse_dir)
    # create a plotter for the output and pass to the eval processor
    plotter = TrajectoryPlotter(fontsize=20)
    evaluator = DenseToSparseEvaluation(plotter=plotter)
    evaluator.fit_linear_model_dense_sparse(dir_dense=dense_dir, dir_sparse=sparse_dir)
