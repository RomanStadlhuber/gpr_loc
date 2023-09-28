# Mobile Robot Localization using Gaussian Process Regression

This mono-repo contains a set of ROS and non-ROS packages that can be used to simulate, train and evaluate a localization scenario that uses Gaussian Process Regression.

Have a look at the documentation of the individual packages `gpr_loc_bringup` and `gpmcl_py` for detailed information on their usage. You can find a rough description below.

## `gpr_loc_bringup` and `linear_ctrl`

This package makes use of the [`taurob_simulation`](https://github.com/TW-Robotics/taurob_simulation) Docker-Environment to start and orchestrate a simulation of a search and rescue (SAR) scenario.

Make sure to put this repository into the `catkin_ws/src` of the docker environment and use it from therein. Although the package is using the ROS setup provided by `taurob_simulation`, its functionality is self-contained and should not require any modification of the docker environment besides installation of the necessary dependencies (see [gpr_loc_bringup/README.md](gpr_loc_bringup/README.md)).

The `linear_ctrl` package was used to generate a simple trajectory in an earlier setup, but this is mentioned in the bringup packages readme as well. Have a look at the configuration `.yaml` and `.launch` files therein to learn how to configure this package.

## `gpmcl_py`

This is a set of development packages, mainly written in Python that are used to generate datasets to train and evaluate Gaussian Processes as well as running inference on Rosbag recordings once these models are available.

Have a look at this packages [README.md](gpmcl_py/README.md) and also its [Example Tutorial](gpmcl_py/EXAMPLE.md) to see how to install and use these modules.

Also have a look at the modules inside the `gpmcl` folder, which contains a useful set of libraries that can be built and extended upon, such as

- GP motion models.
- Landmark observation model with automatic differentiation.
- Classical and Improved Rao-Blackwellized Particle Filter SLAM
- Helper types for interacting with Gaussian Process datasets and models.
- Helper types for processing and visualizing 3D point cloud data.

The modules also provide inline documentation in the form of Markdonw-Formatted Docstrings.
