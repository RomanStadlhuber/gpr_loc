from gp_scenario import GPScenario
import argparse
import pathlib
import GPy
import sys
import os

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
arg_parser.add_argument(
    "-m",
    "--models",
    type=str,
    dest="model_dir",
    metavar="Pretrained models",
    help="The directory containing optimized models",
)
# if the dataset is to be inspected (i.e. printing information)
arg_parser.add_argument(
    "-i",
    "--inspect_only",
    dest="inspect_only",
    default=False,
    action="store_true",
)
# name of the joined datasets (that is, the regression job)
arg_parser.add_argument("-n", "--name", dest="name", required=True, type=str)
# model export directory
arg_parser.add_argument("-o", "--out_dir", dest="out_dir", required=False, type=str)


if __name__ == "__main__":
    # load cli arguments
    args = arg_parser.parse_args()
    train_dirs = [pathlib.Path(d) for d in args.train_dirs]
    test_dir = pathlib.Path(args.test_dir) if args.test_dir else None
    inspect_only: bool = args.inspect_only or False
    scenario_name: str = args.name
    export_dir = pathlib.Path(args.out_dir) if args.out_dir else None
    model_dir = pathlib.Path(args.model_dir) if args.model_dir else None
    # use the plotly backend for graphs
    GPy.plotting.change_plotting_library("plotly")

    regressor = GPScenario(
        scenario_name=scenario_name,
        train_dirs=train_dirs,
        test_dir=test_dir,
        kernel_dir=model_dir,
        inspect_only=inspect_only,
    )

    if inspect_only:
        sys.exit()

    if export_dir is not None:
        if not export_dir.exists():
            os.mkdir(export_dir)
        regressor.export_model_parameters(export_dir)

        if test_dir is not None:
            D_regr = regressor.perform_regression(messages=True)
            if D_regr is not None:
                D_regr.export(export_dir, dataset_name=D_regr.name)
