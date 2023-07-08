from gpmcl.gp_scenario import GPScenario
from typing import Optional
from datetime import datetime
import argparse
import pathlib
import GPy
import sys
import os

arg_parser = argparse.ArgumentParser(
    prog="regression",
    description="Train and/or evaluate a set of dense or sparse Gaussian Processes.",
)

# the first positional parameter is a list of all training dataset directories
arg_parser.add_argument(
    "-d",
    "--data_dirs",
    dest="train_dirs",
    type=str,
    metavar="dataset-dir.",
    help="Directories containing the training datasets.",
    nargs="+",
)
# an optional parameter loads the test dataset and performs regression on it
arg_parser.add_argument(
    "-t",
    "--test",
    dest="test_dir",
    type=str,
    metavar="dataset-dir.",
    help="Directory containing the test dataset.",
)
arg_parser.add_argument(
    "-m",
    "--models",
    type=str,
    dest="model_dir",
    metavar="Pretrained models",
    help="The directory containing optimized models.",
)
# if the dataset is to be inspected (i.e. printing information)
arg_parser.add_argument(
    "-i",
    "--inspect_only",
    dest="inspect_only",
    default=False,
    action="store_true",
    help="Only print information about the training and test dataset.",
)
arg_parser.add_argument(
    "--sparsity",
    dest="sparsity",
    type=int,
    required=False,
    metavar="No. of inducing inputs (> 0) used to train a sparse GP.",
    help="The inducing inputs picked for training will be exported with the model.",
)
# name of the joined datasets (that is, the regression job)
arg_parser.add_argument("-n", "--name", dest="name", type=str)
# model export directory
arg_parser.add_argument("-o", "--out_dir", dest="out_dir", required=False, type=str)


if __name__ == "__main__":
    # load cli arguments
    args = arg_parser.parse_args()
    train_dirs = [pathlib.Path(d) for d in args.train_dirs] if args.train_dirs else None
    test_dir = pathlib.Path(args.test_dir) if args.test_dir else None
    inspect_only: bool = args.inspect_only or False
    scenario_name: str = args.name or datetime.today().strftime("%Y%m%d-%H%M%S")
    export_dir = pathlib.Path(args.out_dir) if args.out_dir else None
    model_dir = pathlib.Path(args.model_dir) if args.model_dir else None
    sparsity: Optional[int] = args.sparsity
    # use the plotly backend for graphs
    GPy.plotting.change_plotting_library("plotly")

    regressor = GPScenario(
        scenario_name=scenario_name,
        train_dirs=train_dirs,
        test_dir=test_dir,
        modelset_dir=model_dir,
        inspect_only=inspect_only,
        sparsity=sparsity,
    )

    if inspect_only:
        sys.exit()

    if export_dir is not None:
        if not export_dir.exists():
            os.mkdir(export_dir)

        if model_dir is None:  # only export models if they do not exist
            regressor.export_models(export_dir, scenario_name)

        if test_dir is not None:
            D_regr = regressor.perform_regression(messages=True)
            if D_regr is not None:
                D_regr.export(export_dir, dataset_name=D_regr.name)
