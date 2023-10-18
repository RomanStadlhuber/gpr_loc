# Training and using Gaussian Process Motion Model

This document describes the full process of training and using a gaussian process motion model in a SLAM task.

Have a look at `bags/demo.bag`, which will be used for this demonstration.

## Installing the Dependencies

Since the dependencies of this repository require exact versioning, it is recommended to use a [Virtual Environment (venv)](https://docs.python.org/3/library/venv.html) inside this folder. Make sure you have the `python3-venv` module or similar installed and run

```bash
# create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate
# update the package installer to the latest version
# this will be needed to install some of the dependencies
pip install --upgrade pip
# now install the dependencies
pip install -r requirements.txt
```

Also, create the following directories:

```bash
mkdir bag models data
```

then download the [demo Rosbag](https://cloud.technikum-wien.at/s/ojEdiPzapAWBZgB) and put it into the `bags` folder.

Now you're all set. If later on you decide you do not want to use the virtual environment any longer, you can exit it using the command `$ deactivate` and remove the entire folder using `$ rm -rf .venv/`.

**Note:** due to some of the dependencies having been updated since the time of development, warnings may occur when running the scripts. These can be safely ignored, for example

> warning in stationary: failed to import cython module: falling back to numpy </br> warning in choleskies: failed to import cython module: falling back to numpy

have no impact on the numerical functionality.

## Generating a Training Dataset

In this section, the command line interface (CLI) of `gpdatagen.py` will be used to generate a dataset that allows to train a gaussian process.

At first, it is a good idea to have a look at the help message provided by the utility.

```bash
python3 gpdatagen.py -h
```

Now, use the below command to generate a dataset.

```bash
python3 gpdatagen.py \
/odom \
--labels /ground_truth/odom \
--bag ./bags/demo.bag \
--out_dir ./data/demo_dataset \
--name demo_dataset \
--time_increment_label \
--deltas
```

Where the last two flags are specific to our purpose

- `--time_increment_label` or `-T` will have the features at timestep `T` relate to the labels at timestep `T+1`.
- `--deltas` ensures that only relative motions are being computed i.e. the features and labels are irrespective of their position in the global frame.

Have a look at `postprocessors.py` to find out how post-processing of generated data works and how to write your own postprocessor!

You will receive output similar to what is below.

```log
Postprocessing feature dataframe.
100%|████████████████████████████████████████████████| 2436/2436 [00:05<00:00, 468.83it/s]
Postprocessing label dataframe.
100%|████████████████████████████████████████████████| 2436/2436 [00:04<00:00, 608.19it/s]
generated datasets successfully
```

## Training a Gaussian Process

This section outlines how to use the generated dataset to train a gaussian process motion model. The CLI of `regression.py` will be used for inspection, training and evaluation. At first, have a look at its help message.

```bash
python3 regression.py -h
```

### Inspecting the Dataset

Before a GP is trained, it is a good idea to have a look at the data.

```bash
python3 regriossion.py \
--data_dirs ./data/demo_dataset/ \
--inspect_only
```

It is also worth to note at this point, that you can pass multiple dataset directories using `--dataset_dirs` or `-d`, as they will be merged internally. This can be useful when intending to train from,multiple datasets at once.

### Performing a Training Run

Use the following command to train a dense GP motion model.

```bash
python3 regression.py \
--data_dirs ./data/demo_dataset \
--name motion_dense \
--out_dir ./models/motion_dense
```

Or alternatively, to train a sparse GP motion model with a number of - for example - ten inducing inputs, run.

```bash
python3 regression.py \
--data_dirs ./data/demo_dataset \
--name motion_sparse \
--out_dir ./models/motion_sparse \
--sparsity 10
```

The training process will then automatically pick the inducing inputs using `kmeans++` clustering.

In the following sections, the commands presented will refer to the dense motion model, but there is virtually no difference.

### Inspecting the Trained Model

The training process will generate an output folder containing the relevant data to load and use a fully trained GP out of the box.

- The kernel hyperparameter files `.npy`. Note that non-permissible characters such as `/` are replaced by `--`.
- The training dataset folder - this is required to recover the exact covariance matrix.
- A `metadata.yaml` file

#### Dense GP

```
motion_dense/
├── delta2d.x (--ground_truth--odom).npy
├── delta2d.y (--ground_truth--odom).npy
├── delta2d.yaw (--ground_truth--odom).npy
├── metadata.yaml
└── motion_dense_dataset
    ├── motion_dense - training__motion_dense_dataset_features.csv
    └── motion_dense - training__motion_dense_dataset_labels.csv
```

#### Sparse GP

Sparse GPs additionally contain a file describing the inducing inputs, which are used to create the approximated covariance instead.

```
motion_sparse/
├── delta2d.x (--ground_truth--odom).npy
├── delta2d.y (--ground_truth--odom).npy
├── delta2d.yaw (--ground_truth--odom).npy
├── metadata.yaml
├── motion_sparse_dataset
│   ├── motion_sparse - training__motion_sparse_dataset_features.csv
│   └── motion_sparse - training__motion_sparse_dataset_labels.csv
└── motion_sparse_inducing_inputs.csv
```

#### The Metadata File

This file describes the location of the relevant input data and hyperparameter files and is used when loading the GP

```yaml
inducing_inputs: null # alternatively, path/to/inputs.csv
models:
  - label: delta2d.x (/ground_truth/odom)
    param_file: delta2d.x (--ground_truth--odom).npy
  - label: delta2d.y (/ground_truth/odom)
    param_file: delta2d.y (--ground_truth--odom).npy
  - label: delta2d.yaw (/ground_truth/odom)
param_file: delta2d.yaw (--ground_truth--odom).npy
set_name: motion_dense
training_data: motion_dense_dataset
```

### Evaluate The Gaussian Process

By using the CLI of `regression.py`, a GP can also be compared against a test-dataset. In this example, the training data will be used just so the required commands are conveyed.

```bash
python3 regression.py \
--models ./models/motion_dense \
--test ./data/demo_dataset
```

Optionally, the `--out_dir` command can be used to store the evaluation data in the same format as the datasets discussed earlier.

## Running SLAM Pipeline

To run the SLAM pipeline on a rosbag recording, use the configuration and CLI provided by `gpmapping_bag.py`. At first, have a look at the help message.

```bash
python3 gpmapping_bag.py -h
```

### Configuring the Pipeline

To make use of the GP motion model in the SLAM pipeline, the pipeline needs to be configured to read the relevant topics from the bag, load the GP model and tune relevant parameters accordingly.

Create a file called `config/slam.yaml` that containg the following values in no relevant order.
You can also have a look at the [example configuration file](config/gpmcl_config_example.yaml) as a general starting point.

#### 3D Scan Mapping Parameters

This module defines the parameters for keypoint detection. To learn more about them, visit the Open3D Documentation on [Intrinsic Shape Signature Keypoints](http://www.open3d.org/docs/latest/tutorial/Advanced/iss_keypoint_detector.html).

- The voxel size is used to downsample the Pointcloud (PCD) for faster processing.
- The `scan_tf` defines the relative orientation (quaternion!) and translation of the 3D Scanner w.r.t. the base frame. You can find these values by inspecting your `.urdf` files (for simulated systems) or performing sensor-vehicle calibration on a real system.
- `min_height` and `max_height` are used to remove outlier keypoints that may appear at the border of the 3D scan.

```yaml
mapper:
  scatter_radius: 1.0
  nms_radius: 1.0
  eig_ratio_32: 0.3
  eig_ratio_21: 0.3
  min_neighbor_count: 10
  voxel_size: 0.05
  scan_tf:
    position: [-0.3, 0, 0.25] # position in base frame
    orientation: [0, 0, 0, 1] # orientation in base frame
  min_height: 0.0
  max_height: 2.0
```

#### Particle Filter SLAM

This module configures the behavior of the Rao-Blackwellized Particle Filter (RBPF) SLAM system, which is at the core of this project.

- Set `improve_proposal_distribution: True` to make use of enhanced RBPF-SLAM (recommended unless you have a _very_ accurate GP).
- `max_unobserved_count` removes (spurious) landmarks that go unobserved too often, however relocalization turned out to fail early on, so it is recommended to make this value as low as possible to keep all landmarks.
- `particle_count` determines the constant number of particles used during the SLAM process, enhanced RBPF SLAM can deal well with a lower amount of particles
- The values of `keypoint_covariance` and `observation_covariance` are the 9 elements of the `3x3` matrices determining the initial positional uncertainty as well as the range-bearing uncertainty of observed keypoints respectively. These values must amount to PSD matrices.
- Since particles perform correspondence checks uniquely, `kdtree_search_radius` is part of this module. A higher value will increase the chances of relocalization at the cost of inaccuracy.
- To amplify the noise behavior of _overconfident_ mean estimates made by the motion model, the noise applied to the partice is increased by separate gain factors in that order: `[x, y, heading]`.

```yaml
fast_slam:
  improve_proposal_distribution: True # if true, algorithm will run FastSLAM 2.0
  max_feature_range: 25 # [m]
  max_unobserved_count: -5000 # remove landmark after unobserved this often
  particle_count: 20
  keypoint_covariance: [0.6, 0, 0, 0, 0.6, 0, 0, 0, 0.6]
  observation_covariance: [0.3, 0, 0, 0, 0.3, 0, 0, 0, 0.3]
  kdtree_search_radius: 0.5
  motion_noise_gain: [10, 10, 10]
```

#### The GP Motion Model

Such that the SLAM pipeline can load the Motion model for inference based on the estimated odometry, it requires the

- `model_dir` which defines the full (or relative to `gpmcl_py`) path to the GP motion model trained above.
- `estimated_motion_labels` and `estimated_twist_labels`, which are the header values of the `*_features.csv` file of the training dataset, which you should pick out by yourself

```yaml
motion_model_gp:
  model_dir: /path/to/models/motion_dense
  estimated_motion_labels:
    x: delta2d.x (/odom)
    y: delta2d.y (/odom)
    theta: delta2d.yaw (/odom)
  estimated_twist_labels:
    x: twist2d.x (/odom)
    y: twist2d.y (/odom)
    theta: twist2d.ang (/odom)
```

#### Rosbag Playback

To load the Rosbag, then read and synchronize the correct topics, the `bag_runner` module.

- `bag_path` defines the full (or relative to `gpmcl_py`) path to the rosbag file.
- `estimated_odometry_topic` will be used to compute the estimated motion and twist which serve as inputs for GP motion regression.
- `pointcloud_topic` is the full name of the [`sensor_msgs/PointCloud2`](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html) topic that describes the raw 3D PCD.
- `groundtruth_topic` this is optional, and as such can be `null`, or the full name of the topic providing the groundtruth pose as `nav_msgs/Odometry`.
- Since messages are synchronized in order to be related to one-another at a specific timestep `sync_period` determines the maximum temporal distance between these messages
- If you want the inference processes to end after a select number of steps (for example, when testing) set `stop_after` accordingly, otherwise just leave it to `-1`

```yaml
bag_runner:
  bag_path: /path/to/bags/demo.bag
  estimated_odometry_topic: /odom # estimated odometry
  pointcloud_topic: /velodyne_points # 3D scan
  groundtruth_topic: /ground_truth/odom # ground truth odometry
  sync_period: 0.1 # [sec.]
  stop_after: -1
```

### Running SLAM inference

Run the following command.

```yaml
python3 gpmapping_bag.py \
./path/to/config/slam.yaml \
--out_dir data/trajectories
```

This will perform SLAM inference and - when done - output the results into the `out_dir` in the form of CSV files.

#### Debugging

Use the flag `-d` or `--dbg_vis`, which will visualize the scan and map state of the most likely particle in an interactive 3D window at each timestep.

Additionally, you can save the pointclouds at each timestep on disk to a desired folder, use the `-pcd` or `--pointcloud_dir` option, passing the path to the containing folder along with it.

#### Plotting

You can use `generate_plots.py` to generate 2D plots of the SLAM trajectory and effective weigths.
A 3D visualization of the merged poinclouds can be generated from exported pcd files using `map3d_post.py`.
**Note** that these features do not provide a CLI as of yet and need to be modified manually.
