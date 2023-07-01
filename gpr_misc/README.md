# Development Package for GPR-Localization

This package includes the code that is used to develop the gaussian process regression scheme for mobile robot localization.

## Before you start

Install the latest version of pip. This is **required** in order to install the required version of Open3D.

```bash
pip install --upgrade pip
```

### Installing all Dependencies

Install the virtualenv module, then create a virtual environment, upgrade PIP and install all dependencies.

```bash
# create and load venv
sudo apt install -y -q python3-virtualenv
python3 -m venv .venv
source .venv/bin/activate
# update to latest PIP
pip install --upgrade pip
# install dependencies
pip install -r requirements.txt
```

#### Installing Open3Ds Development Version

> **Remark**: This might change with future releases of `open3d-cpu>=0.17.0`.

Currently _(July 2023)_ the `gpmcl.mapper.Mapper` class makes use of a method that is not available in the release version of Open3D. In order to get the mapping code to work, you need to install the development version of Open3D using `install_open3d_dev.sh`

## Generating Datasets from Rosbags

The following will give an overview of how to generate and interpret GP datasets from resbags.

GP datasets come in the form `D = (X, y)`, where `X` is a matrix of row vectors that represent the input features for the GP and `y` is the row-vector of scalar-valued output labels.

The `RosbagEncoder` class allows to generate GP datasets in the form of indexed and column-named CSV files (that can be processed using the Pandas library) based on a singel rosbag input.

### Creating Feature Encoders

The parser file uses feature encoding functions that convert ROS messages into feature vectors, with each feature (see `GPFeature` in `helper_types.py`) having both a `column_name` for the exported dataframe and a scalar `value`.

The `rosbags` package provides typings for deserialized ROS messages. In order to create a feature encoder for a specific message type, first import the necessary type

```python
from rosbags.typesys.types import (
    # other imported types ...
    geometry_msgs__msg__Twist as Twist # ... for example, import the Twist type
)
```

and define an encoding function that takes as arguments the ROS message and the topic name (this is used to discern from multiple topics that have the same type). Note that it is **important to include the topic name in the feature column name** as the dataframe can be malformed otherwise!

```python
def __encode_twist(msg: Twist, topic: str) -> List[GPFeature]:
    return __to_gp_features(
        [
            # NOTE: the topic names are integrated into the column name!
            (f"twist.lin.x ({topic})", msg.linear.x),
            (f"twist.lin.y ({topic})", msg.linear.y),
            (f"twist.ang.z ({topic})", msg.angular.z),
        ]
    )

```

Finally, include the your custom encoder function in the encoder map, which is a dictionary that uses the message type name as a key.

```python
ENCODER_MAP = {
    # other encoders go here as well ...
    Twist.__msgtype__: __encode_twist
}
```

### Adding Custom Postprocessing Routines

Have a look at the `postprocessors.py` script. Postprocessors inherit from an abstract base class and can be passed to an instance when calling `RosbagEncoder.encode_bag()` using the `postproc=...` paramater. See `gpdatagen.py` for more details.

### Running the Encoder

The `gpdatagen` utility allows to run the encoder on different rosbags using various settings from the command-line.
To learn more about the arguments, run `gpdatagen -h` to obtain a help message. The below example demonstrates the use of the utility, where all arguments except `--out_dir` are required.

#### Example: Generate a Process Dataset

In the process dataset, the timesteps of the labels are incremented w.r.t. that of the features. To achieve this behavior, use the `-T` or `--time_increment_label` flag.

To compute the deltas between odometry poses, pass the `--deltas` flag to the utility, otherwise the poses will be used.

**Quick Tip**: omit the `--deltas` flag to generate the poses that can be used for plotting trajectiories.

```bash
# generate a process dataset of the form:
# features: state and control-input at time k
# labels: state at time k+1
python3 gpdatagen.py \
/ground_truth/odom /taurob_tracker/cmd_vel_raw \
--label /ground_truth/odom \
--deltas \
--time_increment_label \
--bag path/to/your.bag \
--out_dir ./data \
--name process
```

where the second line lists the names of all the topics that are used for feature generation and the `--label` flag dictates the _list of topics_ that is used to generate labels. Point the encoder to a rosbag file using the `--bag` option and set an output directory where the dataset files can be found vie the `--out_dir` flag. Use the `--name` option to give a label to the dataset so it can be identified at a later point.

#### Example: Generate an Observation Dataset

```bash
# generate an observation dataset of the form:
# features: current state at time k
# labels: observations (sensor data) at time k
python3 gpdatagen.py \
/ground_truth/odom \
--label /odom /taurob_tracker/imu/data \
--deltas \
--bag path/to/your.bag \
--out_dir ./data \
--name observation
```

Note that here only a single topic is used to generate the features and multiple sensor topics are used as labels. Moreover, since observations are instant, there is no timestep increment required.

## Performing Regression on Datasets

The `regression.py` CLI utility allows to create regression models from training datasets, load existing pre-trained models and perform prediction on test-datasets. Run `python3 regression.py -h` for more information on its parameters.

### Recommended folder structure

In the following, three folders containing GP datasets are depicted. The below examples will use two datasets to generate training data and run a test on the third test dataset. It is important to remark that, when loading datasets via the regression utility

- only one dataset (`*_features.csv` & `*_labels.csv`) per folder is allowed
- multiple folders can be loaded and joined into a single trainin dataset
- the **column names** of all datasets **need to be equivalent for both test and training data**

```
data
├── process_test
│   ├── taurob_eval-deltas__process_features.csv
│   └── taurob_eval-deltas__process_labels.csv
├── process_train_ccw
│   ├── taurob_ccw-deltas__process_features.csv
│   └── taurob_ccw-deltas__process_labels.csv
└── process_train_cw
    ├── taurob_cw-deltas__process_features.csv
    └── taurob_cw-deltas__process_labels.csv
```

### Inspecting Datasets (Training & Test)

The `regression.py` script allows to inspect the loaded datasets. I.e. using the data described above, run

```bash
python3 regression.py data/process_train_ccw data/process_train_cw/ --test data/process_test/ --inspect_only
```

which will merely output information about the datasets and exit immediately. Optionally, give the regression scenario a name to identify it at a later point (which will be done in the next examples).

### Generating and Exporting Models

If no directory containing pre-trained models is provided, the utility will generate them automatically. The models can then be saved on disk using the `--out_dir` option.

```bash
# create new models and perform prediction
python3 regression.py \
data/process_train_ccw data/process_train_cw/ \
--name example-scenario \
--out_dir data/models
```

This will run `GPy`s optimization routines and generate a folder named after the `--name` value inside the `--out_dir` directory. This folder will contain the optimized model parameters and regression results (if data was provided using the `--test` flag).

**NOTE**: the exported model parameters will be named after the column names of the labels dataframe. _However_, since ros topics include the `"/"` characters in their names, which is illegal in files, it is replaced with `"--"` (and correspondlingy undone when reloading the models). **Do not modify the filenames of the exported models parameters**. Future versions might include and make use of metadata to migitate this issue.

#### Generate a sparse GP

Add the `--sparsity` flag with a value greater than zero and less than the number of datapoints in your training set. E.g. `--sparsity 10` will construct a sparse GP subset with 10 inducing inputs.

### Reloading exported Models

The `--models` flag can be used to pass the directory that contains the pre-trained models (such as those exported in the example above). If models are provided, the regression utlity will skip creating and optimizing models and load the existing ones instead.

Prediction can then be made using the provided `--test` dataset and saved using the `--export` flag.

```bash
# load existing models and perform prediction
python3 regression.py \
data/process_train_ccw data/process_train_cw/ \
--test data/process_test/ \
--models data/models/example-scenario/ \
--name example-scenario-pretrained \
--out_dir data/models
```

This example will use the pre-trained models from above and only perfrom prediction on the new data. The results of that prediction process will be stored in a new folder named after `--name`, withing the `--out_dir`.

#### Loading a Sparse Regression Model

As of now, a sparse model cannot be detected automatically. Therefore, when loading and evaluating a sparse regression model, specify the `--sparsity=<NUM_INDUCING>` flag in the above command, where `<NUM_INDUCING>` is the exact number of inducing inputs that were used when training the sparse GP .

Below is an example on how to load a sparse model using `helper_types.GPModel`:

```python
>>> from helper_types import GPModel, GPDataset
# choose some example test dataset and make sure it exists
>>> dataset_dir = pathlib.Path("data/process_test")
>>> dataset_dir.exists()
True
>>> D = GPDataset.load(dataset_dir)
# load the test input features
>>> X = D.get_X()
# load some test label data
>>> Y = D.get_Y("your label name")
>>> params_file = pathlib.Path("data/models/sparse-pretrained/example.npy")
# make sure the directory exists...
>>> params_file.exists()
True
>>> sparse_models = GPModel.load_regression_model(
    file=params_file,
    X=X,
    Y=Y,
    sparse=True
)
# print the model
>>> print(model)
```

#### Misceral Information

Internally, the provided datasets are standard-scaled before training and rescaled upon inference. **Do not provide standard-scaled datasets on your own**, as the regression utility will store the scalers when loading the datasets and use them to rescale once inference is complete.

## Running the Localization Pipeline (GPMCL)

The **G**aussan **P**process **M**onte **C**arlo **L**ocalization pipeline uses the pre-trained GPs defined above in conjunction with a 3D Laser Scanner (`sensor_msgs/PointCloud2`) to localize a mobile robot in the plane.

Given a pre-trained GPs for both the process- and optionally the observation model, a localization pipeline can be run on a ROS1 bag file.

```bash
python3 gpmcl_pipeline.py path/to/config.yaml
```

Where `/path/to/config.yaml` points to a file containing the necessary configuration for the pipeline. See `./config/gpmcl_config_example.yaml` for the required format.
