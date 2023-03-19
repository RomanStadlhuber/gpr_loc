#!/bin/bash
rosbag record \
-o taurob_ccw.bag \
/clock \
/taurob_tracker/cmd_vel_raw \
/taurob_tracker/imu/data \
/odom \
/ground_truth/odom
