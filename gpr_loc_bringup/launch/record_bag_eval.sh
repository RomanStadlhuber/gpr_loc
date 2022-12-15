#!/bin/bash
rosbag record \
-o taurob_eval.bag \
/clock \
/taurob_tracker/cmd_vel_raw \
/taurob_tracker/imu/data \
/odom \
/ground_truth/odom