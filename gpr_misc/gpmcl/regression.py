from helper_types import GPDataset, LabelledModel
from dataclasses import dataclass
from typing import List, NamedTuple
import pandas as pd
import numpy as np
import pathlib


@dataclass
class GPRegressionConfig:
    # the path to the directory holding the models
    model_dir: pathlib.Path
    # the path to the models training data (required for loading)
    train_data_path: pathlib.Path
    # if the regression uses a sparse GP
    is_sparse: bool
    # labels (in order) for last state change data
    labels_dX_last: List[str]
    # labels (in order) for estimated state change "control command"
    labels_dU: List[str]


class Prediction(NamedTuple):
    predicted: np.ndarray
    change: np.ndarray


class GPRegression:
    def __init__(self, config: GPRegressionConfig) -> None:
        self.config = config
        # load the training dataset (required for loading the model from GPy)
        self.D_train = GPDataset.load(config.train_data_path)
        (
            self.train_feature_scaler,
            self.train_label_scaler,
        ) = self.D_train.standard_scale()
        # load the models with their labels
        self.labelled_models = LabelledModel.load_labelled_models(
            model_dir=config.model_dir,
            D_train=self.D_train,
            sparse=config.is_sparse,
        )

    def predict(self, X: np.ndarray, dX_last: np.ndarray, dU: np.ndarray) -> Prediction:
        """Predict the next state(s).

        Convert the input data into a `helper_types.GPDataset`.
        The obtained regression data is then converted back into a matrix
        and added to the current state(s) `X`.

        `X` can be a single vector (e.g. `(3 x 1)`) in case of a Kalman filter
        or an entire matrix (e.g. `(3 x M)`) in case of a particle filter.

        Returns the predicted state(s) `X`.
        """
        df_dX_last = pd.DataFrame(data=dX_last, columns=self.config.labels_dX_last)
        df_dU = pd.DataFrame(data=dU, columns=self.config.labels_dU)
        # join the datasets to obtain the feature dataset
        df_X_in = df_dX_last.join(df_dU)
        df_Y = pd.DataFrame()  # empty label dataset (not needed)
        D_in = GPDataset(name="anonymous (GPMCL test dataset)", features=df_X_in, labels=df_Y)
        # perform regression on the dataset
        D_pred = self.__regression(D_in)
        dX_pred = D_pred.labels.to_numpy()
        # add the predicted state change to the current state
        X_pred = X + dX_pred
        # return both the predicted state and the change
        return Prediction(predicted=X_pred, change=dX_pred)

    def __regression(self, D_test: GPDataset) -> GPDataset:
        """Perform predicton on a dataset without labels.

        This function assumes that `D.Y` is empty and ignores it.
        """

        X_test = D_test.get_X()
        regression_labels = pd.DataFrame()
        for labelled_model in self.labelled_models:
            label = labelled_model.label
            model = labelled_model.model
            # perform regression, then rescale and export
            # TODO: export covariance
            (Y_regr, _) = model.predict_noiseless(X_test)
            # create a dataframe for this label and join with the rest of the labels
            df_Y_regr = pd.DataFrame(columns=[label], data=Y_regr)
            regression_labels = pd.concat([regression_labels, df_Y_regr], axis=1)

        D_regr = GPDataset(
            name="anonymous (GPMCL regression)",
            features=D_test.features,
            labels=regression_labels,
        )
        # rescale the output data to its original size
        D_regr.rescale(self.train_feature_scaler, self.train_label_scaler)
        return D_regr
