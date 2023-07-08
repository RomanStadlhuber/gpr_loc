from typing import TypeVar, Callable, Any, List, Tuple, Iterable, Optional, Union, Dict, TypedDict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from pprint import pformat
import pandas as pd
import numpy as np
import pathlib
import yaml
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
        order_by: Optional["GPDataset"] = None,
    ) -> "GPDataset":
        """Uses the filename prefix to load features and labels and construct a new Dataset

        ### Remark

        Pass another dataset `order_by` to force the same column order.
        This requries that the datasets have at least the same columns.
        """

        if file_prefix_name is not None:
            features_dir = dataset_folder.joinpath(f"{file_prefix_name}_features.csv")
            labels_dir = dataset_folder.joinpath(f"{file_prefix_name}_labels.csv")

            if features_dir.exists() and features_dir.is_file() and labels_dir.exists() and labels_dir.is_file():
                name = name or file_prefix_name
                features = pd.read_csv(features_dir, index_col=0)
                labels = pd.read_csv(labels_dir, index_col=0)
                if order_by is None:
                    return GPDataset(name, features, labels)
                else:
                    return GPDataset(name, features[order_by.features.columns], labels[order_by.labels.columns])

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
            if order_by is None:
                return GPDataset(name, feature_df, label_df)
            else:
                return GPDataset(name, feature_df[order_by.features.columns], label_df[order_by.labels.columns])

    @staticmethod
    def join(others: Iterable["GPDataset"], name: str = "joined") -> "GPDataset":
        """join multiple GP Datasets (in the order as passed)"""
        joined_features = pd.concat([other.features for other in others])
        joined_labels = pd.concat([other.labels for other in others])
        return GPDataset(name=name, features=joined_features, labels=joined_labels)

    def check_in_bounds(self, other: "GPDataset") -> Tuple[bool, np.ndarray]:
        """Compute whether `other.features` lie within the trained bounds.

        Returns a tuple of the form `(all_in_bounds, individual)`
        where `individual` lists the result of the bounds check on a per-input basis.

        ### Example

        This example shows the output for inputs of shape `(N_rows, 6)`.

        ```
        >>> X = np.random.rand(2, 6) # get random input
        >>> X /= np.linalg.norm(X) # normalize the input
        >>> D_test(features=X, labels=np.zeros((2,3)))
        >>> all_in_bounds, individual = D_train.check_in_bounds(D_test)
        >>> all_in_bounds
        False
        >>> individual
        array([[1, 1 , 1, 0, 1, 0], [1 ,1 ,1 ,1 ,1 ,1]])
        ```
        """
        X = self.get_X()
        x_min = X.min(axis=0)  # lower training bound
        x_max = X.max(axis=0)  # upper training bound
        X_other = other.get_X()  # features of the other datasets
        # perform the boundary check
        X_in_bounds = np.where((X_other >= x_min) & (X_other <= x_max), 1, 0)
        return bool(np.all(X_in_bounds)), X_in_bounds

    def get_X(self) -> np.ndarray:
        """obtain a column-vector matrix of the dataset features"""
        return self.features.to_numpy()

    def get_Y(self, colname: str) -> np.ndarray:
        """obtain a single column vector for all labels of a column"""
        return self.labels[colname].to_numpy().reshape(-1, 1)

    def export(self, folder: pathlib.Path, dataset_name: str) -> None:
        """export the dataset features and labels to CSV files"""
        if not folder.exists():
            folder.mkdir()
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

        # at first, fit the scalers if none were provided
        if scalers is None:
            feature_scaler.fit(self.features[self.features.columns])
            if not self.labels.empty:
                label_scaler.fit(self.labels[self.labels.columns])

        self.features[self.features.columns] = feature_scaler.transform(self.features[self.features.columns])
        # only apply scaling to labels if they exist
        # this is done because test datasets might not have label values
        if not self.labels.empty:
            self.labels[self.labels.columns] = label_scaler.transform(self.labels[self.labels.columns])
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

    def save(self, dir: pathlib.Path) -> str:
        # TODO: find out why this won't work with the full name
        filename = f"""{self.name.replace("/", "--")}.npy"""
        np.save(dir / filename, self.parameters)
        return filename

    @staticmethod
    def load(file: pathlib.Path) -> "GPModel":
        """load a model wrapper instance from a file"""
        name = file.name.split(".")[0].replace("--", "/")
        return GPModel(name, np.load(file))

    @staticmethod
    def load_regression_model(
        file: pathlib.Path,
        X: np.ndarray,
        Y: np.ndarray,
        inducing_inputs: Optional[pd.DataFrame],
    ) -> Union[GPy.models.GPRegression, GPy.models.SparseGPRegression]:
        """load a regression model from a file"""
        _, dim, *__ = X.shape
        # create an ARD kernel (with a diagonal lengthscale matrix, that is) in order to
        # properly load the model parameters
        rbf_kernel = GPy.kern.RBF(input_dim=dim, ARD=True)
        m_load = (
            GPy.models.GPRegression(X, Y, initialize=False, kernel=rbf_kernel)
            if inducing_inputs is None
            else GPy.models.SparseGPRegression(
                X,
                Y,
                initialize=False,
                kernel=rbf_kernel,
                Z=inducing_inputs.to_numpy() if inducing_inputs is not None else None,
            )
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
                """Loaded parameter dimenions to not match required parameter dimensions.

This error likely occurs because
    - you are trying to load a sparse model into a dense one or vice versa
    - the version of GPy used to create the model does not match the version loading it

Please see also the arguments of GPModel.load_regression_model.
            """
            )


@dataclass
class LabelledModel:
    """A wrapper class for associating a `GPy.Model` with a feature label."""

    label: str
    model: Union[GPy.models.GPRegression, GPy.models.SparseGPRegression]

    def save_params(self, dir: pathlib.Path) -> pathlib.Path:
        file_path = dir / f"""{self.label.replace("/", "--")}.npy"""
        np.save(file_path, self.model.param_array)
        return file_path

    @staticmethod
    def load_labelled_models(
        model_dir: pathlib.Path, D_train: GPDataset, sparsity: Optional[int]
    ) -> List["LabelledModel"]:
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
            model = GPModel.load_regression_model(kernel_file, X, Y, sparsity=sparsity)
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


class GPModelMetadata(TypedDict):
    label: str
    param_file: str


class GPModelSetMetadata(TypedDict):
    """Metadata wrapper for a set of models used for common inference."""

    # list of all the models and their labels
    models: List[GPModelMetadata]
    # folder name of the dataset used to train the model
    training_data: str
    # name of the inducing input file (if any)
    inducing_inputs: Optional[str]
    # name of the entire set
    set_name: str


@dataclass
class GPModelSet:
    """A wrapper about labelled, pre-loaded GP models.

    Contains the training dataset and various other relevant data.
    """

    # the pre-configured models
    gp_models: List[LabelledModel]
    # the scaled training dataset
    D_train: GPDataset
    # inducing inputs used (in case of a sparse GP)
    inducing_inputs: Optional[pd.DataFrame]
    # scaler to training data feature scale
    training_feature_scaler: StandardScaler
    # scaler to training data label scale
    training_label_scaler: StandardScaler
    # a constant remark that should come up in all metadata files
    remark: str = "Inducing inputs are already std.-scaled acc. to the training data."

    def perform_regression(
        self,
        D_test_unscaled: Optional[GPDataset] = None,
        D_test_scaled: Optional[GPDataset] = None,
        name: Optional[str] = None,
    ) -> GPDataset:
        if D_test_unscaled is not None:
            D_test_unscaled.standard_scale(scalers=(self.training_feature_scaler, self.training_label_scaler))
        D_test = D_test_scaled or D_test_unscaled
        if D_test is None:
            raise ValueError("Unable to perform regression, dataset does not exist!")
        # reorder input features according to the training data
        D_test.features = D_test.features[self.D_train.features.columns]
        X_test = D_test.get_X()
        regression_labels = pd.DataFrame()
        for labelled_model in self.gp_models:
            label = labelled_model.label
            model = labelled_model.model
            # perform regression, then rescale and export
            # TODO: export covariance
            (Y_regr, _) = model.predict_noiseless(X_test)
            # create a dataframe for this label and join with the rest of the labels
            df_Y_regr = pd.DataFrame(columns=[label], data=Y_regr)
            regression_labels = pd.concat([regression_labels, df_Y_regr], axis=1)

        D_regr = GPDataset(
            name=name,
            features=D_test.features,
            labels=regression_labels,
        )
        D_regr.rescale(self.training_feature_scaler, self.training_label_scaler)
        return D_regr

    @staticmethod
    def export_models(
        labelled_models: List[LabelledModel],
        dataset: GPDataset,
        inducing_inputs: Optional[pd.DataFrame],
        root_folder: pathlib.Path,
        name: str,
    ) -> pathlib.Path:
        """Export a set of models to the disk.

        Export generates a folder containing
        - training dataset
        - the model kernel hyperparameter files
        - inducing input parameter files (if any)
        - a file listing the metadata used to quickly load saved
        """
        if not root_folder.exists():
            root_folder.mkdir()
        # save the models and store their metadata
        models_metadata = list(
            map(
                lambda labelled_model: GPModelMetadata(
                    label=labelled_model.label, param_file=labelled_model.save_params(root_folder).name
                ),
                labelled_models,
            )
        )
        # save the dataset
        dataset_dir_name = f"{name}_dataset"
        dataset.export(root_folder / dataset_dir_name, dataset_dir_name)
        # export inducing inputs if they exist
        if inducing_inputs is not None:
            filename_inducing = f"{name}_inducing_inputs.csv"
            inducing_inputs.to_csv(root_folder / filename_inducing)
            # write metadata with inducing inputs
            with open(root_folder / "metadata.yaml", "w") as f_metadata:
                yaml.safe_dump(
                    data=GPModelSetMetadata(
                        models=models_metadata,
                        training_data=dataset_dir_name,
                        inducing_inputs=filename_inducing,
                        set_name=name,
                    ),
                    stream=f_metadata,
                )
        else:
            # write metadata without inducing inputs
            with open(root_folder / "metadata.yaml", "w") as f_metadata:
                yaml.safe_dump(
                    data=GPModelSetMetadata(
                        models=models_metadata, training_data=dataset_dir_name, inducing_inputs=None, set_name=name
                    ),
                    stream=f_metadata,
                )

        return root_folder

    @staticmethod
    def load_models(root_folder: pathlib.Path) -> "GPModelSet":
        if not root_folder.exists():
            raise FileNotFoundError(f"Unable to locate GP Model folder '{root_folder}'.")
        elif not (root_folder / "metadata.yaml").exists():
            raise FileNotFoundError(f"'{root_folder}' does not contain a file called 'metadata.yaml'.")
        with open(root_folder / "metadata.yaml", "r") as f_metadata:
            metadata: GPModelSetMetadata = yaml.safe_load(stream=f_metadata)
            D_train = GPDataset.load(
                dataset_folder=root_folder / metadata["training_data"], name=f"""{metadata["set_name"]}_training_data"""
            )
            (feature_scaler, label_scaler) = D_train.standard_scale()
            inducing_inputs = (
                pd.read_csv(root_folder / metadata["inducing_inputs"], index_col=0)
                if metadata["inducing_inputs"] is not None
                else None
            )
            gp_models = list(
                map(
                    lambda model_metadata: LabelledModel(
                        label=model_metadata["label"],
                        model=GPModel.load_regression_model(
                            root_folder / model_metadata["param_file"],
                            X=D_train.get_X(),
                            Y=D_train.get_Y(model_metadata["label"]),
                            # the inducing inputs, reordered to fit the training data order
                            # this is a safeguard, but the order will probably be right most of the time
                            inducing_inputs=inducing_inputs[D_train.features.columns] if inducing_inputs else None,
                        ),
                    ),
                    metadata["models"],
                )
            )
            return GPModelSet(
                gp_models=gp_models,
                D_train=D_train,
                inducing_inputs=inducing_inputs,
                training_feature_scaler=feature_scaler,
                training_label_scaler=label_scaler,
            )
