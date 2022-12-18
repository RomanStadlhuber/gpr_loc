#!/usr/bin/env python3


"""The command-line utility that wraps the Rosbag Parser and allows to export GP Datasets from rosbags."""

from rosbag_encoder import RosbagEncoder
from typing import List
import argparse
import pathlib


arg_parser = argparse.ArgumentParser(
    prog="gpdatagen",
    description="utility used to generate GP datasets from rosbags",
)

arg_parser.add_argument(
    "feature_topics",
    metavar="Input Feature Topics",
    type=str,
    help="Topics to use as GP input features",
    nargs="+",  # at least one, arbitrarly many and throws error if none supplied
)
arg_parser.add_argument(
    "--label", dest="label_topics", required=True, type=str, nargs="+"
)
arg_parser.add_argument("--bag", dest="bagfile_path", required=True, type=str)
arg_parser.add_argument(
    "-o",
    "--out_dir",
    dest="out_dir",
    metavar="Output files directory",
    type=str,
    help="Directory in which the output files are generated",
    default=".",
)

arg_parser.add_argument(
    "-n",
    "--name",
    dest="dataset_name",
    metavar="Name of the dataset",
    type=str,
    help="Name of the datset (e.g. 'process' or 'observation'),",
    required=True,
)


if __name__ == "__main__":
    # parse the cli args
    args = arg_parser.parse_args()
    feature_topics: List[str] = args.feature_topics
    label_topic: List[str] = args.label_topics
    rosbag_path = pathlib.Path(args.bagfile_path)
    out_dir = pathlib.Path(args.out_dir or ".")
    dataset_name: str = args.dataset_name
    # encode the rosbag
    reader = RosbagEncoder(bagfile_path=rosbag_path)
    dataset = reader.encode_bag(feature_topics, label_topic)
    if dataset is not None:
        dataset.export(out_dir, dataset_name)
        print("generated datasets successfully")
