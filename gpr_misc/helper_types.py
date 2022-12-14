from typing import TypeVar, Callable, Any, List
from dataclasses import dataclass
import pandas as pd

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

    # features are multidimensional and each index of a feature vector stands for some separate information
    features: pd.DataFrame
    # labels are scalars and as such all labels only represent a single column vector
    labels: pd.Series


# function typing used to encode messages into GPR feature vectors
FeatureEncodeFn = Callable[[TRosMsg], List[GPFeature]]
