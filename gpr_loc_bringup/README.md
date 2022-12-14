# `gpr_loc_bringup`

The bringup-package for the gazebo simulation that is used to generate training and test datasets for gaussian process regression. <br/>
This package intends to
- modify the Taurob Gazebo model to publish [ground-truth odometry](https://classic.gazebosim.org/tutorials?tut=ros_gzplugins#P3D(3DPositionInterfaceforGroundTruth))
- loads the default `empty.world` provided by the `gazebo_ros` package
- predefines trajectories used to generate datasets
- configure a linear controller to enable traversing of the trajectories
    - see the `linear_controller` ros-package for more details

## Launching the simulation

All of the functionality listed above is invoked using a single launch file.

```bash
roslaunch gpr_loc_bringup bringup.launch
```

To set the direction of the training trajectory, supply the parameter `dir_ccw:=false` (defaults to `true`).

## Recording Rosbags to generate Datasets

Invoke the `launch/record_bag_*.sh` scripts to record the required data from the running simulation. The recorded data includes

- ground truth odometry - `nav_msgs/Odometry`
- pre-computed mechanical odometry - `nav_msgs/Odometry`
- velocity control signals - `geometry_msgs/Twist`
- raw IMU data - `sensor_msgs/IMU`
