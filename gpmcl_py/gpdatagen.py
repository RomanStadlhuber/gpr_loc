#!/usr/bin/env python3


"""The command-line utility that wraps the Rosbag Parser and allows to export GP Datasets from rosbags."""

from gpmcl.rosbag_encoder import RosbagEncoder
from gpmcl.postprocessors import OdomDeltaPostprocessor
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
arg_parser.add_argument("--label", dest="label_topics", required=True, type=str, nargs="+")
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

arg_parser.add_argument(
    "-T",
    "--time_increment_label",
    dest="time_increment_label",
    help="Set this if the features are provided at timestep k and the labels generated at k+1",
    default=False,
    action="store_true",
)

# TODO: implement this arument for optional postprocessing
arg_parser.add_argument(
    "--deltas",
    dest="deltas",
    help="Whether or not to compute odom deltas",
    default=False,
    action="store_true",
)

if __name__ == "__main__":
    # parse the cli args
    args = arg_parser.parse_args()
    feature_topics: List[str] = args.feature_topics
    label_topic: List[str] = args.label_topics
    rosbag_path = pathlib.Path(args.bagfile_path)
    out_dir = pathlib.Path(args.out_dir or ".")
    dataset_name: str = args.dataset_name
    time_increment_label: bool = args.time_increment_label
    compute_deltas: bool = args.deltas or False
    # create the postprocessor
    # TODO: make this code adaptive by adding a flag for the postprocessor name
    proc = OdomDeltaPostprocessor(odom_topics={"/ground_truth/odom", "/odom"}) if compute_deltas else None
    # encode the rosbag
    reader = RosbagEncoder(bagfile_path=rosbag_path)
    dataset = reader.encode_bag(
        feature_topics,
        label_topic,
        time_increment_on_label=time_increment_label,
        postproc=proc,
    )
    if dataset is not None:
        if not out_dir.exists():
            out_dir.mkdir()
        dataset.export(out_dir, dataset_name)
        print("generated datasets successfully")
