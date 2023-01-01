from typing import TypeVar, Callable, Any, List, Tuple, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from pprint import pformat
import pandas as pd
import numpy as np
import pathlib

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
    def load(dataset_folder: pathlib.Path, data_file_prefix_name: str) -> "GPDataset":
        """Uses the filename prefix to load features and labels and construct a new Dataset"""

        features_dir = dataset_folder.joinpath(f"{data_file_prefix_name}_features.csv")
        labels_dir = dataset_folder.joinpath(f"{data_file_prefix_name}_labels.csv")

        if (
            features_dir.exists()
            and features_dir.is_file()
            and labels_dir.exists()
            and labels_dir.is_file()
        ):
            name = data_file_prefix_name
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

    def get_X(self) -> np.ndarray:
        """obtain a column-vector matrix of the dataset features"""
        return self.features.to_numpy()

    def get_Y(self, colname: str) -> np.ndarray:
        """obtain a single column vector for all labels of a column"""
        return self.labels[colname].to_numpy().reshape(-1, 1)

    def export(self, folder: pathlib.Path, dataset_name: str) -> None:
        """export the dataset features and labels to CSV files"""
        self.features.to_csv(
            pathlib.Path.joinpath(folder, f"{self.name}__{dataset_name}_features.csv")
        )
        self.labels.to_csv(
            pathlib.Path.joinpath(folder, f"{self.name}__{dataset_name}_labels.csv")
        )

    def standard_scale(self) -> Tuple[StandardScaler, StandardScaler]:
        """standard scale this dataset

        returns the fitted `sklearn.StandardScaler` which can then be used to rescale the data
        """
        feature_scaler = StandardScaler()
        label_scaler = StandardScaler()
        self.features[self.features.columns] = feature_scaler.fit_transform(
            self.features[self.features.columns]
        )
        self.labels[self.labels.columns] = label_scaler.fit_transform(
            self.labels[self.labels.columns]
        )
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
        self.labels[self.labels.columns] = fitted_label_scaler.inverse_transform(
            self.labels[self.labels.columns]
        )

    def join(self, other: Type["GPDataset"]) -> "GPDataset":
        """join this dataset with another dataset"""
        joined_features = self.features.join(other.features)
        joined_labels = self.labels.join(other.labels)
        joined_name = f"({self.name}) x ({other.name})"

        return GPDataset(
            name=joined_name, features=joined_features, labels=joined_labels
        )

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
