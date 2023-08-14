from gpmcl.helper_types import GPDataset, DatasetPostprocessor
from typing import Tuple, Set, List
from tqdm import tqdm
import pandas as pd
import numpy as np

# a shorthand handy for postprocessing the deltas
Pose2D = Tuple[float, float, float]


class OdomDeltaPostprocessor(DatasetPostprocessor):
    """The postprocessor that converts odometry poses in dataframes to deltas.

    It reduces a dataframe of n rows to (n-1) rows because a delta is generated
    for each i-th and (i+1)-th row.

    As input, the class requires the list of all odometry topics on which to
    apply conversion.
    """

    def __init__(self, odom_topics: Set[str]) -> None:
        self.odom_topics = odom_topics
        super().__init__()

    def postprocess_dataset(self, dataset: GPDataset) -> GPDataset:
        """Transform a dataset containing 2D pose components into deltas

        NOTE: in order for this to work, some strict datastructure needs to be assumed.
        See the `pose_from_topic` function for more information.
        """
        print("Postprocessing feature dataframe.")
        new_features = self.convert_dataframe(dataset.features, labels=False)
        print("Postprocessing label dataframe.")
        new_labels = self.convert_dataframe(dataset.labels, labels=True)
        return GPDataset(
            name=f"{dataset.name}-deltas",
            features=new_features,
            labels=new_labels,
        )

    @staticmethod
    def compute_pose_delta(frm: Pose2D, to: Pose2D) -> Pose2D:
        """Computes the geometrically correct delta between 2D poses.

        Poses come in the form (x, y, yaw).
        Returns a pose in the form (x, y, yaw).
        """
        x_frm, y_frm, yaw_frm = frm
        # the inverse of the rotation matrix of the starting pose
        R_from_inv = np.array(
            [
                [np.cos(yaw_frm), np.sin(yaw_frm)],
                [-np.sin(yaw_frm), np.cos(yaw_frm)],
            ]
        )
        # the inverse matrix of the starting pose, which has the form
        # [R^-1 | R^-1 * -(x,y)]
        T_from_inv = np.block(
            [
                [R_from_inv, R_from_inv @ -np.array([[x_frm], [y_frm]])],
                [np.array([0, 0, 1])],
            ]
        )
        x_to, y_to, yaw_to = to
        T_to = np.array(
            [
                [np.cos(yaw_to), -np.sin(yaw_to), x_to],
                [np.sin(yaw_to), np.cos(yaw_to), y_to],
                [0, 0, 1],
            ]
        )
        # compute the delta pose as homogeneous matrix
        T_delta = T_from_inv @ T_to
        # compute the 2D rotation frmm the y and x components of the R mat using atan2d
        sin_yaw = T_delta[1, 0]
        cos_yaw = T_delta[0, 0]
        yaw_delta = np.arctan2(sin_yaw, cos_yaw)  # x = atan2d(y=sin(x), x=cos(x))
        x_delta = T_delta[0, 2]
        y_delta = T_delta[1, 2]
        return (x_delta, y_delta, yaw_delta)

    @staticmethod
    def transform_twist_into_frame(twist: np.ndarray, frame: Pose2D) -> np.ndarray:
        """Transform `twist` into `frame`."""
        _, _, theta = frame
        _, _, w = twist
        Rmat_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        dx, dy = Rmat_inv @ twist[:2].reshape((-1, 1))
        # return the transformed x and y twists, omega (w) stays the same
        return np.array([dx, dy, w], dtype=np.float64)

    def convert_dataframe(self, df: pd.DataFrame, labels: bool = False) -> pd.DataFrame:
        """Wrapper function to apply the postprocessing logic on an entire dataframe

        This method treats the label-dataframe differently.
        It only keeps the motion deltas but discards the twists.
        Use `labels=True` to indicate that a dataframe contains label data and needs its twists removed.
        """

        row_count, *_ = df.shape

        old_columns: List[str] = []
        modified_colums: List[str] = []
        twist_columns: List[str] = []
        for topic in self.odom_topics:
            if f"pose2d.x ({topic})" not in df.columns:
                continue

            twist_columns.extend(
                [
                    f"twist2d.x ({topic})",
                    f"twist2d.y ({topic})",
                    f"twist2d.ang ({topic})",
                ]
            )

            old_columns.extend(
                [
                    f"pose2d.x ({topic})",
                    f"pose2d.y ({topic})",
                    f"pose2d.yaw ({topic})",
                    *twist_columns,
                ]
            )
            modified_colums.extend(
                [
                    f"delta2d.x ({topic})",
                    f"delta2d.y ({topic})",
                    f"delta2d.yaw ({topic})",
                    *twist_columns,
                ]
            )
        # the columns of the original dataframe that will not be modified
        remaining_columns = set(df.columns.to_list()).difference(old_columns)

        # the new dataframe that will be iteratively filled in this function
        new_df = (
            pd.DataFrame(columns=[*modified_colums, *remaining_columns])
            if not labels
            else pd.DataFrame(columns=modified_colums)
        )

        def pose_from_topic(row: pd.Series, topic: str) -> Pose2D:
            """retreives the 2D pose of a topic assuming valid column notation

            NOTE: if the column notation is not correct this function will fail and throw an error
            """
            try:
                x, y, yaw = row[  # type: ignore
                    [
                        f"pose2d.x ({topic})",
                        f"pose2d.y ({topic})",
                        f"pose2d.yaw ({topic})",
                    ]
                ]
                return (x, y, yaw)
            except Exception:
                raise KeyError(f"No 2D pose can be extracted from this topics data: '{topic}'")

        def twist_from_topic(row: pd.Series, topic: str) -> np.ndarray:
            try:
                dx, dy, dtheta = row[  # type: ignore
                    [
                        f"twist2d.x ({topic})",
                        f"twist2d.y ({topic})",
                        f"twist2d.ang ({topic})",
                    ]
                ]
                return np.array([dx, dy, dtheta])
            except Exception:
                raise KeyError(f"No 2D twist can be extracted from this topics data: '{topic}'")

        for i in tqdm(range(row_count - 2)):
            row_1st = df.loc[i]  # the current row
            row_2nd = df.loc[i + 1]  # the next row
            # compute the 2d deltas for the following odometry poses
            for odom_topic in self.odom_topics:
                if f"pose2d.x ({odom_topic})" not in df.columns:
                    continue
                pose_1st = pose_from_topic(row_1st, odom_topic)
                pose_2nd = pose_from_topic(row_2nd, odom_topic)
                pose_delta = OdomDeltaPostprocessor.compute_pose_delta(frm=pose_1st, to=pose_2nd)
                delta_x, delta_y, delta_yaw = pose_delta
                new_df.loc[i, f"delta2d.x ({odom_topic})"] = delta_x
                new_df.loc[i, f"delta2d.y ({odom_topic})"] = delta_y
                new_df.loc[i, f"delta2d.yaw ({odom_topic})"] = delta_yaw
                if not labels:
                    twist_2nd = twist_from_topic(row_2nd, odom_topic)
                    # TODO: should it really be in the previous frame?
                    dx, dy, w = OdomDeltaPostprocessor.transform_twist_into_frame(twist_2nd, pose_1st)
                    new_df.loc[i, f"twist2d.x ({odom_topic})"] = dx
                    new_df.loc[i, f"twist2d.y ({odom_topic})"] = dy
                    new_df.loc[i, f"twist2d.ang ({odom_topic})"] = w
            new_df.loc[i, list(remaining_columns)] = row_1st[list(remaining_columns)]
        if labels:
            new_df_twist_columns = [col for col in df.columns if "twist2d" in col]
            new_df = new_df.drop(columns=new_df_twist_columns)
        return new_df
