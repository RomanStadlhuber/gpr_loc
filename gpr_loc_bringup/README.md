# `gpr_loc_bringup`

The bringup-package for the gazebo simulation that is used to generate training and test datasets for gaussian process regression. <br/>
This package intends to
- modify the Taurob Gazebo model to publish [ground-truth odometry](https://classic.gazebosim.org/tutorials?tut=ros_gzplugins#P3D(3DPositionInterfaceforGroundTruth))
- loads the default `empty.world` provided by the `gazebo_ros` package
- predefines trajectories used to generate datasets
- configure a linear controller to enable traversing of the trajectories
    - see the `linear_controller` ros-package for more details
- disable the `robot_pose_ekf` package invoked in `taurob_tracker_control.launch` and use groundtruth odometry instead
    - the ground truth odometry is published using the `p3d_base_controller` (see `taurob_tacker_flipper.xacro`)
    - the `p3d_tf_broadcaster_node` echoes the groundtruth to the TF tree

## Install Dependencies

**TODO:** add a `.rosinstall` file.

### Prebuilt ROS Packages

```bash
sudo apt install -y -q \
    ros-$ROS_DISTRO-move-base ros-$ROS_DISTRO-gmapping ros-$ROS_DISTRO-explore-lite
```

### External Dependencies

#### Laser Scan Merging

In order to run `gmapping` on both laser scans, they're combined using the `scan_merger` node.

Install and build the [`ira_laser_tools`](https://github.com/iralabdisco/ira_laser_tools) package.

#### Linear Controller

This package also depends on the `linear_controller` package, which is provided in the root of the `gpr_loc` repository.

## Launching the simulation

All of the functionality listed above is invoked using a single launch file.

```bash
roslaunch gpr_loc_bringup bringup.launch
```

To set the direction of the training trajectory, supply the parameter `dir_ccw:=false` (defaults to `true`).

### Disabling the linear Controller

```bash
roslaunch gpr_loc_bringup bringup.launch linear_ctrl:=false
```

### Launching the Search And Rescue (SAR) World

**NOTE** It is recommended to disable the controller for these worlds and use teleop instead
(see the optional argument).

```bash
roslaunch gpr_loc_bringup bringup.launch sar_world:=true [linear_ctrl:=false]
```
### Starting Gazebo in GUI Mode

To any invocation of `bringup.launch`, add `gui:=true`.

## SLAM and Exploration

A separate launch file exists for SLAM (using either manual navigation or [`explore_lite`](http://wiki.ros.org/explore_lite)). To start the SLAM simulation scenario, run

```bash
roslaunch gpr_loc_bringup bringup_mapping.launch
```

which will start a simulation in a maze-like world, enabling the robot to navigate using [`move_base`](http://wiki.ros.org/explore_lite) and create an occupancy grid map using [`gmapping`](http://wiki.ros.org/gmapping?distro=noetic).

## Recording Rosbags to generate Datasets

Invoke the `launch/record_bag_*.sh` scripts to record the required data from the running simulation. The recorded data includes

- the simulation time - `std_msgs/Time`
- ground truth odometry - `nav_msgs/Odometry`
- pre-computed mechanical odometry - `nav_msgs/Odometry`
- velocity control signals - `geometry_msgs/Twist`
- raw IMU data - `sensor_msgs/IMU`
