from typing import Optional, Iterable, List
from helper_types import GPDataset, GPModel
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import pathlib
import GPy


@dataclass
class LabelledModel:
    label: str
    model: GPy.Model


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
        train_dirs: Iterable[pathlib.Path],
        test_dir: Optional[pathlib.Path] = None,
        kernel_dir: Optional[pathlib.Path] = None,
        inspect_only: bool = False,
        sparsity: Optional[float] = None,
    ) -> None:
        # the name of the regression scenario
        self.scenario = scenario_name
        # the directories containing the data and the models
        self.train_dirs = train_dirs
        self.test_dir = test_dir
        self.kernel_dir = kernel_dir
        # the training and test datasets
        training_datasets = self.load_datasets(self.train_dirs)
        self.D_train = GPDataset.join(training_datasets, f"{self.scenario} - training")
        # scale the training dataset
        (
            self.train_feature_scaler,
            self.train_label_scaler,
        ) = self.D_train.standard_scale()
        self.D_test = (
            GPDataset.load(dataset_folder=self.test_dir) if self.test_dir else None
        )
        # scale the test dataset if it exists
        (self.test_feature_scaler, self.test_label_scaler,) = (
            self.D_test.standard_scale() if self.D_test else (None, None)
        )
        # the names of the columns used in the regression process
        self.labels = self.D_train.labels.columns
        # a list containing the optimized model kernels
        self.models: List[LabelledModel] = []

        self.D_train.print_info()
        if self.D_test:
            self.D_test.print_info()

        # warn the user of the sparsity is at an incorrect rate
        if sparsity and (sparsity >= 1.0 or sparsity <= 0.0):
            raise Exception(
                f"""Incorrect sparsity level ({sparsity})!
    The sparsity level needs to be within a range of ]0.0, 1.0[ (i.e. not included).
    Based on this level, a sample count relative to that the full dataset will be picked.
            """
            )

        # the relative number of samples (between 0 and 1) or nothing
        self.sparsity = sparsity if sparsity and 0.0 < sparsity <= 1.0 else None

        if inspect_only:
            return
        # generate models if none were provided, otherwise load them
        if self.kernel_dir is None:
            self.generate_models(messages=True)
        else:
            self.load_kernels()

    def load_datasets(self, dirs: Iterable[pathlib.Path]) -> Iterable[GPDataset]:
        """load all datasets from a list of directories"""
        for dir in dirs:
            if not dir.exists() or not dir.is_dir():
                raise FileExistsError(
                    f"The dataset directory '{str(dir)}' does not exist."
                )
        # load all datasets from the subdirectories
        datasets = [GPDataset.load(d) for d in dirs]
        return datasets

    def generate_models(
        self, messages: bool = False, export_directory: Optional[pathlib.Path] = None
    ) -> Iterable[GPy.Model]:
        """Generate models for each label in the training data"""
        # load the input data for all models
        X = self.D_train.get_X()
        # obtain dimensionality of input data
        num_elements, dim, *__ = X.shape
        # list used to store all models
        models: List[GPy.Model] = []
        # individually create models for each label_
        for label in self.labels:
            Y = self.D_train.get_Y(label)
            # define the kernel function for the GP
            rbf_kernel = GPy.kern.RBF(input_dim=dim, variance=1.0, lengthscale=1.0)
            if self.sparsity is not None:
                print(
                    f"Constructing a sparse GP at a relative sample count of {int(self.sparsity*100)} %"
                )
            # build the model
            # TODO: do we need to load existing kernel hyperparamters from dense GP to help the optimizater?
            model = (
                GPy.models.GPRegression(X, Y, rbf_kernel)
                if self.sparsity is None
                else GPy.models.SparseGPRegression(
                    X, Y, rbf_kernel, num_inducing=int(self.sparsity * num_elements)
                )
            )
            if messages:
                # print information about the model
                print(model)
                # start optimizing the model
                print(f"Beginning model optimization for label {label}...")
            model.optimize(messages=messages)
            models.append(LabelledModel(label, model))
            if (
                export_directory
                and export_directory.exists()
                and export_directory.is_dir()
            ):
                # NOTE: it is paramount that this filename is equal to the label column
                model_metadata = GPModel(name=label, parameters=model.param_array)
                model_metadata.save(export_directory)

        self.models = models
        return models

    def load_kernels(self) -> None:
        """Load all kernels from the kernel directory"""
        # skip if there aren't any kernels to load
        if (
            self.kernel_dir is None
            or not self.kernel_dir.exists()
            or not self.kernel_dir.is_dir()
        ):
            return

        X = self.D_train.get_X()
        kernel_files = [x for x in self.kernel_dir.iterdir() if x.suffix == ".npy"]
        models: List[GPy.Model] = []
        for kernel_file in kernel_files:
            label = kernel_file.name.split(".npy")[0].replace("--", "/")
            Y = self.D_train.get_Y(label)
            model = GPModel.load_regression_model(kernel_file, X, Y)
            models.append(LabelledModel(label, model))
        # update the models used for regression
        self.models = models

    def perform_regression(self, messages: bool = False) -> Optional[GPDataset]:
        """Perform regression if a test dataset was provided and models are loaded."""
        if self.models is None or self.D_test is None:
            return None

        if messages:
            print(f"performing regression in scenario {self.scenario}..")

        X_test = self.D_test.get_X()
        regression_labels = pd.DataFrame()
        for labelled_model in tqdm(self.models):
            label = labelled_model.label
            model = labelled_model.model
            # perform regression, then rescale and export
            # TODO: export covariance
            (Y_regr, _) = model.predict_noiseless(X_test)
            # create a dataframe for this label and join with the rest of the labels
            df_Y_regr = pd.DataFrame(columns=[label], data=Y_regr)
            regression_labels = pd.concat([regression_labels, df_Y_regr], axis=0)

        D_regr = GPDataset(
            name=f"{self.scenario}_regression-output",
            features=self.D_train.features,
            labels=regression_labels,
        )
        D_regr.rescale(self.train_feature_scaler, self.train_label_scaler)
        return D_regr

    def export_model_parameters(self, dir: pathlib.Path) -> None:
        """export the paramters of all models in this scenario to the target directory"""
        if len(self.models) > 0:
            for model in self.models:
                m = GPModel(name=model.label, parameters=model.model.param_array)
                m.save(dir)
