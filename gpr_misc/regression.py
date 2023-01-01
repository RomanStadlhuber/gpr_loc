from helper_types import GPDataset
import argparse
import pathlib
import GPy

arg_parser = argparse.ArgumentParser(
    prog="regression",
    description="the script used to perform regression on gaussian process data",
)

arg_parser.add_argument(
    dest="dir",
    type=str,
    metavar="Dataset Directory",
    help="the dataset directory",
)
arg_parser.add_argument(
    dest="prefix",
    type=str,
    metavar="Dataset Filename Prefix",
    help="the dataset filename prefix (what's before _features/labels.csv)",
)

"""TODO:

make the program load multiple training and test datasets

- Option 1: accept a dedicated folder with a required structure:
    - root folder name is the scenario
        - contains training and test subfolders
        - these subfolders contain all of the respective datasets
- Option 2: accept one folder for training and one for test
- Option 3: accept mutliple prefixes, seperated into training and test
"""

if __name__ == "__main__":
    args = arg_parser.parse_args()
    datset_dir = pathlib.Path(args.dir)
    dataset_prefix: str = args.prefix
    # use the plotly backend for graphs
    GPy.plotting.change_plotting_library("plotly")
    # load dataset D = (X, Y)
    D = GPDataset.load(dataset_folder=datset_dir, data_file_prefix_name=dataset_prefix)
    # print information about the dataset
    D.print_info()
    # standard-scale the dataset and store the scaler objects
    feature_scaler, label_scaler = D.standard_scale()
    # load input data as pure numpy matrix, transpose so input is a column-vector matrix
    X = D.get_X()
    # define an arbitrary regression target column name
    Y = D.get_Y("delta2d.x (/ground_truth/odom)")
    # obtain dimensionality of input data
    _, dim, *__ = X.shape
    print(
        f"""
    Dataset dimensions:
    X: {X.shape}
    Y: {Y.shape}
    """
    )
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

    # TODO: load a test dataset and perform prediction, see:
    # https://gpy.readthedocs.io/en/deploy/GPy.core.html?highlight=predict#GPy.core.gp.GP.predict
