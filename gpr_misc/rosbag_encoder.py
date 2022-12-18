# from rosbags.rosbag1 import Reader
# from rosbags.serde import deserialize_cdr, ros1_to_cdr
from helper_types import GPFeature, GPDataset
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from rosbags.typesys.types import nav_msgs__msg__Odometry as Odometry
from message_feature_encoders import get_encoder
from typing import List, Optional
import pandas as pd
import pathlib


class RosbagEncoder:
    """helper class used to read rosbags"""

    def __init__(self, bagfile_path: pathlib.Path) -> None:
        self.bagfile_path = bagfile_path
        pass

    def encode_bag(
        self,
        feature_topics: List[str],
        label_topics: List[str],
        time_increment_on_label: bool = False,  # if the label should contain data from the next timestep
        timestamp_min: Optional[int] = None,
        timestamp_max: Optional[int] = None,
    ) -> Optional[GPDataset]:
        """read the rosbag and encode all"""
        if not self.bagfile_path.exists() or not self.bagfile_path.is_file():
            return None

        try:
            with Reader(self.bagfile_path) as reader:

                dataset_feature_df: Optional[pd.DataFrame] = None
                dataset_label_df: Optional[pd.DataFrame] = None

                buffered_features = dict()
                buffered_labels = dict()

                iteration_index = 0

                for connection, _, rawdata in reader.messages():

                    # encode and buffer feature messages
                    if connection.topic in feature_topics:
                        encoder_fn = get_encoder(connection.msgtype)
                        if encoder_fn is None:
                            continue
                        msg = deserialize_cdr(
                            ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                        )
                        features = encoder_fn(msg, connection.topic)
                        buffered_features[connection.topic] = features

                    # encode label message
                    elif connection.topic in label_topics:
                        encoder_fn = get_encoder(connection.msgtype)
                        if encoder_fn is None:
                            continue
                        msg = deserialize_cdr(
                            ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                        )
                        label = encoder_fn(msg, connection.topic)
                        buffered_labels[connection.topic] = label

                    else:
                        continue

                    # once all features and a label have been buffered:
                    # create a dataframe for both features and labels and joint with the total dataframe
                    # fmt: off
                    has_all_features = set(buffered_features.keys()) == set(feature_topics)
                    has_all_labels = set(buffered_labels.keys()) == set(label_topics)
                    # fmt: on

                    if has_all_features and has_all_labels:
                        # flatten all features and labels of this iteration into single lists
                        flat_features: List[GPFeature] = [
                            feature
                            for features in buffered_features.values()
                            for feature in features
                        ]
                        feature_names: List[str] = [
                            feature.column_name for feature in flat_features
                        ]
                        feature_values = [feature.value for feature in flat_features]
                        flat_labels: List[GPFeature] = [
                            label
                            for labels in buffered_labels.values()
                            for label in labels
                        ]
                        label_names = [label.column_name for label in flat_labels]
                        label_values = [label.value for label in flat_labels]

                        feature_df = pd.DataFrame(
                            data=[feature_values],
                            columns=feature_names,
                            index=[iteration_index],
                        )
                        label_df = pd.DataFrame(
                            data=[label_values],
                            columns=label_names,
                            index=[iteration_index],
                        )

                        # append to dataset features
                        if dataset_feature_df is None:
                            dataset_feature_df = feature_df
                        else:
                            dataset_feature_df = pd.concat(
                                [dataset_feature_df, feature_df]
                            )

                        # append to dataset labels
                        if dataset_label_df is None:
                            dataset_label_df = label_df
                        else:
                            dataset_label_df = pd.concat([dataset_label_df, label_df])

                        # reset for next iteration
                        iteration_index += 1
                        buffered_features.clear()
                        buffered_labels.clear()

                bag_name = self.bagfile_path.name.split(".")[0]

                # after done iterating the rosbag
                dataset = GPDataset(
                    name=bag_name, features=dataset_feature_df, labels=dataset_label_df
                )

                return dataset

        except Exception as e:
            print(str(e))
            return None

    def read_trajectory(
        self, odom_topic: str, label: str, bagfile_path: Optional[pathlib.Path] = None
    ) -> Optional[pd.DataFrame]:
        try:
            with Reader(bagfile_path or self.bagfile_path) as reader:
                pos_df = pd.DataFrame(columns=["x", "y", "label"])
                idx = 0
                for connection, _, rawdata in reader.messages():
                    if connection.topic == odom_topic:
                        msg: Odometry = deserialize_cdr(
                            ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                        )
                        pos_df.loc[idx] = [
                            msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            label,
                        ]
                        idx += 1
                return pos_df

        except Exception as e:
            print(str(e))
            return None
