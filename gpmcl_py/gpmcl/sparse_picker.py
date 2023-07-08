from sklearn.cluster import KMeans
from gpmcl.helper_types import GPDataset
import pandas as pd


class SparsePicker:
    """Provides methods for picking inducing inputs for a sparse GP."""

    @staticmethod
    def pick_kmeanspp(D_train_scaled: GPDataset, num_inducing: int) -> pd.DataFrame:
        """Pick `num_inducing` inputs for a sparse gaussian process using kmeans++ method.

        Clusters the data into `num_inducing` partitions and returns their centers.

        ### Returns

        A `pd.DataFrame` with the same columns as found in `D_train_scaled.features`.

        ### Remarks

        Make sure the training data is already scaled, as later test data needs to use the same
        scaler as the feature dataset!
        """
        k_means = KMeans(n_clusters=num_inducing, init="k-means++", n_init="auto")
        k_means = k_means.fit(D_train_scaled.get_X())
        inducing_features = pd.DataFrame(columns=D_train_scaled.features.columns, data=k_means.cluster_centers_)
        return inducing_features
