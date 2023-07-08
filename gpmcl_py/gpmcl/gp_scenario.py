from typing import Optional, Iterable, List, Union
from gpmcl.helper_types import GPDataset, GPModel, LabelledModel, GPModelSet
from gpmcl.sparse_picker import SparsePicker
from dataclasses import dataclass
import pandas as pd
import numpy as np
import pathlib
import GPy


class GPScenario:
    """A class that can be used to construct models and/or perform regression in a target scenario

    A scenario consists of (possibly multiple) training datasets and a sigle test dataset,
    all of which have the same feature and label structure.

    This class allows to generate and optimize the models, export their parameters and load them
    for use at a later time.

    Optimized models can be used for regression, where a dataset is returned consisting of the regression
    inputs and the mean value outputs as labels.

    An important NOTE is that since the ROS topic names contain slashes, these need to be replaced when
    serializing the models into files. In this case, the "/" character is replaced by "--" and vice versa.
    This logic is also found in the helper_types script GPModel class!
    """

    def __init__(
        self,
        scenario_name: str,
        train_dirs: Optional[Iterable[pathlib.Path]] = None,
        test_dir: Optional[pathlib.Path] = None,
        modelset_dir: Optional[pathlib.Path] = None,
        inspect_only: bool = False,
        sparsity: Optional[int] = None,
    ) -> None:
        # the name of the regression scenario
        self.scenario = scenario_name
        # the directories containing the data and the models
        self.train_dirs = train_dirs
        self.test_dir = test_dir
        self.modelset_dir = modelset_dir
        self.load_sparse = sparsity is not None
        if modelset_dir is None and train_dirs is not None:
            self.load_training_set_from_dirs()
        self.load_test_set()
        # the names of the columns used in the regression process
        self.labels = self.D_train.labels.columns
        # a list containing the optimized model kernels
        self.models: List[LabelledModel] = []
        # a buffer for the inducing inputs feature dataframe
        self.inducing_features: Optional[pd.DataFrame] = None

        self.D_train.print_info()
        if self.D_test:
            self.D_test.print_info()

        num_datapoints, *_ = self.D_train.get_X().shape
        # warn the user if the sparsity count is too large
        if sparsity and (sparsity > num_datapoints or sparsity <= 0):
            raise Exception(
                f"""
            Number of sparse datapoints is out of bounds ({sparsity}).
            Sparsity needs to be greater than 0 and less than number of data points ({num_datapoints}).
            Please provide a sparsity between 1 and {num_datapoints}.
            To speed up the search process, 15 or less inducing inputs are recommended.
            """
            )

        # the relative number of samples (between 0 and 1) or nothing
        self.sparsity = sparsity if sparsity and 0 < sparsity <= num_datapoints else None

        if inspect_only:
            return
        # generate models if none were provided, otherwise load them
        if self.modelset_dir is None:
            self.generate_models(messages=True)
        else:
            self.load_models()

    def load_datasets(self, dirs: Iterable[pathlib.Path]) -> Iterable[GPDataset]:
        """load all datasets from a list of directories"""
        for dir in dirs:
            if not dir.exists() or not dir.is_dir():
                raise FileExistsError(f"The dataset directory '{str(dir)}' does not exist.")
        # load all datasets from the subdirectories
        datasets = [GPDataset.load(d) for d in dirs]
        return datasets

    def load_training_set_from_dirs(self) -> None:
        if self.train_dirs is None:
            return
        # the training and test datasets
        training_datasets = self.load_datasets(self.train_dirs)
        self.D_train = GPDataset.join(training_datasets, f"{self.scenario} - training")
        # TODO: check if this truly is a copy operation
        self.D_train_unscaled = self.D_train
        # scale the training dataset
        (
            self.train_feature_scaler,
            self.train_label_scaler,
        ) = self.D_train.standard_scale()

    def load_test_set(self) -> None:
        """Loads the test dataset and standard-scales it according to the training feature and label scalers."""
        self.D_test = GPDataset.load(dataset_folder=self.test_dir) if self.test_dir else None
        # scale the test dataset if it exists
        (
            self.test_feature_scaler,
            self.test_label_scaler,
        ) = (
            self.D_test.standard_scale(scalers=(self.train_feature_scaler, self.train_label_scaler))
            if self.D_test
            else (None, None)
        )

    def generate_models(self, messages: bool = False) -> Iterable[GPy.Model]:
        """Generate models for each label in the training data"""
        # load the input data for all models
        X = self.D_train.get_X()
        # obtain dimensionality of input data
        _, dim, *__ = X.shape
        # compute the sparse inducing inputs once, they're the same for all processes
        if self.sparsity is not None:
            print(f"Constructing a sparse GP with {self.sparsity} inducing inputs.")
            self.inducing_features = SparsePicker.pick_kmeanspp(self.D_train, self.sparsity)
        # list used to store all models
        models: List[GPy.Model] = []
        # individually create models for each label_
        for label in self.labels:
            Y = self.D_train.get_Y(label)
            # define the kernel function for the GP
            rbf_kernel = GPy.kern.RBF(input_dim=dim, variance=1.0, lengthscale=1.0, ARD=True)
            # build the model
            model = (
                GPy.models.GPRegression(X, Y, kernel=rbf_kernel)
                if self.sparsity is None
                else GPy.models.SparseGPRegression(
                    X,
                    Y,
                    kernel=rbf_kernel,
                    # inducing inputs as picked by the Sparse-Picker
                    Z=self.inducing_features.to_numpy(),
                )
            )
            if messages:
                # print information about the model
                print(model)
                # start optimizing the model
                print(f"Beginning model optimization for label {label}...")
            model.optimize(messages=messages)
            models.append(LabelledModel(label, model))

        self.models = models
        return models

    def load_models(self) -> None:
        """Load all kernels from the kernel directory"""
        # skip if there aren't any kernels to load
        modelset = GPModelSet.load_models(self.modelset_dir)
        self.D_train = modelset.D_train
        self.train_feature_scaler = modelset.training_feature_scaler
        self.train_label_scaler = modelset.training_label_scaler
        self.inducing_features = modelset.inducing_inputs
        # load the test set again, but this time with the new scalers
        self.load_test_set()
        # store the models for later use (i.e. inference)
        self.models = modelset.gp_models

    def perform_regression(self, messages: bool = False) -> Optional[GPDataset]:
        """Perform regression if a test dataset was provided and models are loaded."""
        if self.models is None or self.D_test is None:
            return None

        if messages:
            print(f"performing regression in scenario {self.scenario}..")

        X_test = self.D_test.get_X()
        regression_labels = pd.DataFrame()
        for labelled_model in self.models:
            label = labelled_model.label
            model = labelled_model.model
            # perform regression, then rescale and export
            # TODO: export covariance
            (Y_regr, _) = model.predict_noiseless(X_test)
            # create a dataframe for this label and join with the rest of the labels
            df_Y_regr = pd.DataFrame(columns=[label], data=Y_regr)
            regression_labels = pd.concat([regression_labels, df_Y_regr], axis=1)

        D_regr = GPDataset(
            name=f"{self.scenario}_regression-output",
            features=self.D_test.features,
            labels=regression_labels,
        )
        D_regr.rescale(self.train_feature_scaler, self.train_label_scaler)
        return D_regr

    def export_models(self, dir: pathlib.Path, name: str) -> None:
        """export the paramters of all models in this scenario to the target directory"""
        if len(self.models) > 0:
            X = self.D_train.get_X()
            GPModelSet.export_models(
                labelled_models=self.models,
                dataset=self.D_train_unscaled,
                inducing_inputs=self.inducing_features,
                root_folder=dir,
                name=name,
            )
