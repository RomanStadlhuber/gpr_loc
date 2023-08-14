from gpmcl.helper_types import GPModelSet, GPDataset
from gpmcl.config import MotionModelGPConfig
from typing import Tuple
import pandas as pd
import numpy as np
import pathlib


class MotionModel:
    def __init__(self, config: MotionModelGPConfig) -> None:
        self.config = config
        self.colnames_estimated_motion = [
            config["estimated_motion_labels"]["x"],
            config["estimated_motion_labels"]["y"],
            config["estimated_motion_labels"]["theta"],
        ]
        self.colnames_estimated_twist = [
            config["estimated_twist_labels"]["x"],
            config["estimated_twist_labels"]["y"],
            config["estimated_twist_labels"]["theta"],
        ]

        # region: load GP models
        model_path = pathlib.Path(config["model_dir"])
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(
                f"""
Unable to load GP Models!
'{model_path}' does not exist or is not a directory.
            """
            )
        else:
            self.models = GPModelSet.load_models(model_path)
        # endregion: load GP models

    def predict(
        self,
        estimated_motion: np.ndarray,
        estimated_twist: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform motion prediction using Gaussian process regression.

        Returns `mean_vectors, variance_vectors`, where the latter can be used to construct
        diagonal covariance matrices using `np.diag(variance_vector)`.

        ### Remark
        The input values can have dimensions `(3x1)` or `(Nx3)`,
        but they are required to have identical dimensions.
        """
        # create dataframes with the required column names for regression
        features_motion = pd.DataFrame(columns=self.colnames_estimated_motion, data=estimated_motion)
        features_twist = pd.DataFrame(columns=self.colnames_estimated_twist, data=estimated_twist)
        features = features_motion.join(features_twist)
        # create an unnamed dataset without labels
        D_in = GPDataset(name="", features=features, labels=pd.DataFrame())
        # pass the unscaled dataset to the regression method.
        # it will inernally scale the dataset accordingly
        D_regr, df_Var_regr = self.models.perform_regression(D_test_unscaled=D_in)
        # return the predicted motion mean vectors
        return (D_regr.labels.to_numpy(), df_Var_regr.to_numpy())
