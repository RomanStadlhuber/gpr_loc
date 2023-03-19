#!/bin/bash
rosbag record \
-o taurob_cw.bag \
/clock \
/taurob_tracker/cmd_vel_raw \
/taurob_tracker/imu/data \
/odom \
/ground_truth/odom