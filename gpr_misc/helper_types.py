from typing import TypeVar, Callable, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
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

    def export(self, folder: pathlib.Path, dataset_name: str) -> None:
        """export the dataset features and labels to CSV files"""
        self.features.to_csv(
            pathlib.Path.joinpath(folder, f"{self.name}__{dataset_name}_features.csv")
        )
        self.labels.to_csv(
            pathlib.Path.joinpath(folder, f"{self.name}__{dataset_name}_labels.csv")
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
