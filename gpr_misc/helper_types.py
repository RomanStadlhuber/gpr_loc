from typing import TypeVar, Callable, Any, List, Tuple, Iterable, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from pprint import pformat
import pandas as pd
import numpy as np
import pathlib
import GPy

TRosMsg = TypeVar("TRosMsg")


@dataclass
class GPFeature:
    """the class representing a single index in a GP feature vector"""

    # the name of the pandas DataFrame column used to store this feature
    column_name: str
    # basically only some form of number, but any np type is permitted
    value: Any


@dataclass
class GPDataset:
    """Wrapper for an entire GP dataset"""

    # name of the context in which the dataset was created
    name: str
    # features are multidimensional and each index of a feature vector stands for some separate information
    features: pd.DataFrame
    # set of all label vectors generated from a rosbag, needs to be separated into its columns to obtain training data
    labels: pd.DataFrame

    @staticmethod
    def load(
        dataset_folder: pathlib.Path,
        name: Optional[str] = None,
        file_prefix_name: Optional[str] = None,
    ) -> "GPDataset":
        """Uses the filename prefix to load features and labels and construct a new Dataset"""

        if file_prefix_name is not None:
            features_dir = dataset_folder.joinpath(f"{file_prefix_name}_features.csv")
            labels_dir = dataset_folder.joinpath(f"{file_prefix_name}_labels.csv")

            if features_dir.exists() and features_dir.is_file() and labels_dir.exists() and labels_dir.is_file():
                name = name or file_prefix_name
                features = pd.read_csv(features_dir, index_col=0)
                labels = pd.read_csv(labels_dir, index_col=0)
                return GPDataset(name, features, labels)
            else:
                raise FileNotFoundError(
                    f"""One of the dataset files cannot be found.
                    - features: {features_dir}
                    - labels: {labels_dir}
                """
                )
        else:
            feature_files = [x for x in dataset_folder.iterdir() if x.is_file() and "features.csv" in x.name]

            label_files = [x for x in dataset_folder.iterdir() if x.is_file() and "labels.csv" in x.name]
            # check that the directory only contains one dataset
            if len(feature_files) != len(label_files) != 1:
                raise FileNotFoundError(
                    f"""
                    There is more than one dataset in the directory '{dataset_folder.name}'.
                    Only one dataset (_features and _labels files) is allowed per
                    """
                )
            # load the dataset from the directory
            feature_df = pd.read_csv(feature_files[0], index_col=0)
            label_df = pd.read_csv(label_files[0], index_col=0)
            return GPDataset(name=name or dataset_folder.name, features=feature_df, labels=label_df)

    @staticmethod
    def join(others: Iterable["GPDataset"], name: str = "joined") -> "GPDataset":
        """join multiple GP Datasets (in the order as passed)"""
        joined_features = pd.concat([other.features for other in others])
        joined_labels = pd.concat([other.labels for other in others])
        return GPDataset(name=name, features=joined_features, labels=joined_labels)

    def get_X(self) -> np.ndarray:
        """obtain a column-vector matrix of the dataset features"""
        return self.features.to_numpy()

    def get_Y(self, colname: str) -> np.ndarray:
        """obtain a single column vector for all labels of a column"""
        return self.labels[colname].to_numpy().reshape(-1, 1)

    def export(self, folder: pathlib.Path, dataset_name: str) -> None:
        """export the dataset features and labels to CSV files"""
        self.features.to_csv(pathlib.Path.joinpath(folder, f"{self.name}__{dataset_name}_features.csv"))
        self.labels.to_csv(pathlib.Path.joinpath(folder, f"{self.name}__{dataset_name}_labels.csv"))

    def standard_scale(
        self, scalers: Optional[Tuple[StandardScaler, StandardScaler]] = None
    ) -> Tuple[StandardScaler, StandardScaler]:
        """standard scale this dataset

        Returns the fitted `sklearn.StandardScaler` which can then be used to rescale the data.

        Optionally, pass external feature and label scalers in the form `(feature_scaler, label_scaler)`
        (these will then be returned as well).
        """
        feature_scaler = StandardScaler() if scalers is None else scalers[0]
        label_scaler = StandardScaler() if scalers is None else scalers[1]
        self.features[self.features.columns] = feature_scaler.fit_transform(self.features[self.features.columns])
        self.labels[self.labels.columns] = label_scaler.fit_transform(self.labels[self.labels.columns])
        # return the fitted scaler
        return (feature_scaler, label_scaler)

    def rescale(
        self,
        fitted_feature_scaler: StandardScaler,
        fitted_label_scaler: StandardScaler,
    ) -> None:
        """rescale a transformed dataset given a fitted `sklearn.StandardScaler`

        performs in-place rescaling of the dataset
        """
        self.features[self.features.columns] = fitted_feature_scaler.inverse_transform(
            self.features[self.features.columns]
        )
        self.labels[self.labels.columns] = fitted_label_scaler.inverse_transform(self.labels[self.labels.columns])

    def print_info(self) -> None:
        rows, *_ = self.features.shape

        print(
            f"""
---
GP Dataset: "{self.name}"

    - feature columns:
{pformat(self.features.columns.to_list(), indent=8)}

    - label columns:
{pformat(self.labels.columns.to_list(), indent=8)}

    - total vector entries: {rows}
---
"""
        )


@dataclass
class GPModel:
    """Helper class to save and load GPy model data

    The save/load logic is implemented according to:
    https://github.com/SheffieldML/GPy#saving-models-in-a-consistent-way-across-versions
    """

    name: str
    parameters: np.ndarray

    def save(self, dir: pathlib.Path) -> None:
        # TODO: find out why this won't work with the full name
        filename = self.name.replace("/", "--")
        np.save(dir / f"{filename}.npy", self.parameters)

    @staticmethod
    def load(file: pathlib.Path) -> "GPModel":
        """load a model wrapper instance from a file"""
        name = file.name.split(".")[0].replace("--", "/")
        return GPModel(name, np.load(file))

    @staticmethod
    def load_regression_model(
        file: pathlib.Path, X: np.ndarray, Y: np.ndarray, sparse: bool = False
    ) -> Union[GPy.models.GPRegression, GPy.models.SparseGPRegression]:
        """load a regression model from a file"""
        _, dim, *__ = X.shape
        # create an ARD kernel (with a diagonal lengthscale matrix, that is) in order to
        # properly load the model parameters
        rbf_kernel = GPy.kern.RBF(input_dim=dim, ARD=True)
        m_load = (
            GPy.models.GPRegression(X, Y, initialize=False, kernel=rbf_kernel)
            if not sparse
            else GPy.models.SparseGPRegression(X, Y, initialize=False, kernel=rbf_kernel)
        )
        m_load.update_model(False)
        m_load.initialize_parameter()
        model_data = GPModel.load(file)
        try:
            m_load[:] = model_data.parameters
            m_load.update_model(True)
            return m_load
        except ValueError:
            raise Exception(
                """You are trying to load a sparse model into a dense one.
To load a sparse model, pass the optional parameter 'sparse = True' to GPModel.load_regression_model.
            """
            )


@dataclass
class LabelledModel:
    """A wrapper class for associating a `GPy.Model` with a feature label."""

    label: str
    model: Union[GPy.models.GPRegression, GPy.models.SparseGPRegression]

    @staticmethod
    def load_labelled_models(model_dir: pathlib.Path, D_train: GPDataset, sparse: bool) -> List["LabelledModel"]:
        """Load multiple labelled models form a directory.

        Allows to load both dense and sparse models.

        This functions requires the training data in order to load models using `GPy`.
        Please NOTE that the training data labels and model labels need to be the same!
        """

        if not model_dir.exists() or not model_dir.is_dir():
            raise Exception(f"Provided model_dir '{model_dir}' does not exist or is not a path!")

        # load the feature matrix
        X = D_train.get_X()

        kernel_files = [x for x in model_dir.iterdir() if x.suffix == ".npy"]
        models: List[LabelledModel] = []
        for kernel_file in kernel_files:
            label = kernel_file.name.split(".npy")[0].replace("--", "/")
            Y = D_train.get_Y(label)
            # load a dense or sparse regression model (based on the constructors parameter)
            model = GPModel.load_regression_model(kernel_file, X, Y, sparse=sparse)
            models.append(LabelledModel(label, model))

        return models


# function typing used to encode messages into GPR feature vectors
FeatureEncodeFn = Callable[[TRosMsg, str], List[GPFeature]]


class DatasetPostprocessor(ABC):
    """Inherit from this class to apply postprocessing logic to a GP dataset."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def postprocess_dataset(self, dataset: GPDataset) -> GPDataset:
        """convert an existing dataset by postprocessing it"""
        pass
