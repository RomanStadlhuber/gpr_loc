from typing import Iterable
from helper_types import GPDataset
import numpy as np
import argparse
import pathlib
import GPy
import sys

arg_parser = argparse.ArgumentParser(
    prog="regression",
    description="the script used to perform regression on gaussian process data",
)

# the first positional parameter is a list of all training dataset directories
arg_parser.add_argument(
    dest="train_dirs",
    type=str,
    metavar="Training Dataset Directory",
    help="The root directory containing all training datasets",
    nargs="+",
)
# an optional parameter loads the test dataset and performs regression on it
arg_parser.add_argument(
    "-t",
    "--test",
    dest="test_dir",
    type=str,
    metavar="Test Dataset Directory",
    help="The root directory containing the test dataset",
)
# if the dataset is to be inspected (i.e. printing information)
arg_parser.add_argument(
    "-i",
    "--inspect_only",
    dest="inspect_only",
    default=False,
    action="store_true",
)


def load_datasets(dirs: Iterable[pathlib.Path]) -> Iterable[GPDataset]:
    """load all datasets from a list of directories"""
    for dir in dirs:
        if not dir.exists() or not dir.is_dir():
            raise FileExistsError(f"The dataset directory '{str(dir)}' does not exist.")
    # load all datasets from the subdirectories
    datasets = [GPDataset.load(d) for d in dirs]
    return datasets


if __name__ == "__main__":
    # load cli arguments
    args = arg_parser.parse_args()
    train_dirs = [pathlib.Path(d) for d in args.train_dirs]
    test_dir = pathlib.Path(args.test_dir) if args.test_dir else None
    inspect_only: bool = args.inspect_only or False

    # use the plotly backend for graphs
    GPy.plotting.change_plotting_library("plotly")

    # load all training datasets and join them into a single one
    training_datasets = load_datasets(train_dirs)
    D = GPDataset.join(training_datasets, name="taurob process model - training")
    # standard-scale the dataset and store the scaler objects
    feature_scaler, label_scaler = D.standard_scale()
    # load input data as pure numpy matrix, transpose so input is a column-vector matrix
    X = D.get_X()
    # define an arbitrary regression target column name
    Y = D.get_Y("delta2d.x (/ground_truth/odom)")
    # obtain dimensionality of input data
    _, dim, *__ = X.shape

    # print information about the dataset
    D.print_info()
    print(
        f"""
    Dataset dimensions:
    X: {X.shape}
    Y: {Y.shape}
    """
    )

    if inspect_only:
        sys.exit()

    # define the kernel function for the GP
    rbf_kernel = GPy.kern.RBF(input_dim=dim, variance=1.0, lengthscale=1.0)
    # build the model
    model = GPy.models.GPRegression(X, Y, rbf_kernel)
    # print information about the model
    print(model)
    # start optimizing the model
    print("Beginning model optimization...")
    model.optimize(messages=True)

    print(
        f"""
---
Most significant input dimensions: {model.get_most_significant_input_dimensions()}
---
    """
    )

    if test_dir is not None:
        D_test = GPDataset.load(dataset_folder=test_dir)
        test_feature_scaler, test_label_scaler = D_test.standard_scale()
        X_test = D_test.get_X()
        Y_test = D_test.get_Y("delta2d.x (/ground_truth/odom)")
        D_test.print_info()
        (Y_regr, Var_regr) = model.predict_noiseless(X_test)

        print(
            f"""
regression output:
    - Y: {np.shape(Y_regr)}
    - Cov: {np.shape(Var_regr)}

regression metrics:
    - MSE: {np.mean(np.square(Y_regr - Y_test))}
    - variance:
        - min: {np.min(Var_regr)}
        - avg: {np.mean(Var_regr)}
        - max: {np.max(Var_regr)}
        """
        )

    # TODO: load a test dataset and perform prediction, see:
    # https://gpy.readthedocs.io/en/deploy/GPy.core.html?highlight=predict#GPy.core.gp.GP.predict
