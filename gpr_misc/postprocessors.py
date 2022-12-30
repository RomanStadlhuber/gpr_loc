from helper_types import GPDataset, DatasetPostprocessor
from typing import Tuple, Set, List
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
        return GPDataset(
            name=f"{dataset.name}-deltas",
            features=self.convert_dataframe(dataset.features),
            labels=self.convert_dataframe(dataset.labels),
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

    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper function to apply the postprocessing logic on an entire dataframe"""

        row_count, *_ = df.shape

        old_columns: List[str] = []
        modified_colums: List[str] = []
        for topic in self.odom_topics:
            if f"pose2d.x ({topic})" not in df.columns:
                continue

            old_columns.extend(
                [
                    f"pose2d.x ({topic})",
                    f"pose2d.y ({topic})",
                    f"pose2d.yaw ({topic})",
                ]
            )
            modified_colums.extend(
                [
                    f"delta2d.x ({topic})",
                    f"delta2d.y ({topic})",
                    f"delta2d.yaw ({topic})",
                ]
            )
        # the columns of the original dataframe that will not be modified
        remaining_columns = set(df.columns.to_list()).difference(old_columns)

        # the new dataframe that will be iteratively filled in this function
        new_df = pd.DataFrame(columns=[*modified_colums, *remaining_columns])

        def pose_from_topic(row: pd.Series, topic: str) -> Pose2D:
            """retreives the 2D pose of a topic assuming valid column notation

            NOTE: if the column notation is not correct this function will fail and throw an error
            """
            try:
                x, y, yaw = row[
                    [
                        f"pose2d.x ({topic})",
                        f"pose2d.y ({topic})",
                        f"pose2d.yaw ({topic})",
                    ]
                ]
                return (x, y, yaw)
            except Exception:
                raise KeyError(
                    f"No 2D pose can be extracted from this topics data: '{topic}'"
                )

        for i in range(row_count - 2):
            row_1st = df.loc[i]  # the current row
            row_2nd = df.loc[i + 1]  # the next row
            # compute the 2d deltas for the following odometry poses
            for odom_topic in self.odom_topics:
                if f"pose2d.x ({odom_topic})" not in df.columns:
                    continue

                pose_1st = pose_from_topic(row_1st, odom_topic)
                pose_2nd = pose_from_topic(row_2nd, odom_topic)
                pose_delta = OdomDeltaPostprocessor.compute_pose_delta(
                    frm=pose_1st, to=pose_2nd
                )
                delta_x, delta_y, delta_yaw = pose_delta
                new_df.loc[i, f"delta2d.x ({odom_topic})"] = delta_x
                new_df.loc[i, f"delta2d.y ({odom_topic})"] = delta_y
                new_df.loc[i, f"delta2d.yaw ({odom_topic})"] = delta_yaw
            # add the remaining columns aren't modified
            new_df.loc[i, list(remaining_columns)] = row_1st[list(remaining_columns)]

        return new_df
