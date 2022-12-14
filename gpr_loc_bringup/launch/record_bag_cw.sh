#!/bin/bash
rosbag record \
# filename
-o taurob_cw.bag \
# simulation time
/clock
# control signal "u"
/taurob_tracker/cmd_vel_raw \
# raw IMU data
/taurob_tracker/imu/data \
# precomputed mechanical odometry (from gazebo plugin)
/odom \
# ground truth odometry (from gazebo plugin)
/ground_truth/odom \