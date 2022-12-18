from rosbag_encoder import RosbagEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# import pandas as pd
import pathlib

if __name__ == "__main__":
    # the two rosbags used for training data
    ccw_bagfile_path = pathlib.Path("bags/taurob_ccw_2022-12-14-16-40-22.bag")
    cw_bagfile_path = pathlib.Path("bags/taurob_cw_2022-12-15-16-03-39.bag")

    encoder = RosbagEncoder(ccw_bagfile_path)

    ccw_trajectory = encoder.read_trajectory(
        "/ground_truth/odom", "counter-clockwise", bagfile_path=ccw_bagfile_path
    )
    """cw_trajectory = encoder.read_trajectory(
        "/ground_truth/odom", "counter-clockwise", bagfile_path=cw_bagfile_path
    )"""

    # join the training trajectory dataframe for plotting
    # trajectories = pd.concat([cw_trajectory, ccw_trajectory])

    sns.relplot(data=ccw_trajectory, x="x", y="y", hue="label")
    plt.show()
