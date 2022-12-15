# Development Package for GPR-Localization

This package includes the code that is used to develop the gaussian process regression scheme for mobile robot localization.

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

### Running the Encoder

The `gpdatagen` utility allows to run the encoder on different rosbags using various settings from the command-line.
To learn more about the arguments, run `gpdatagen -h` to obtain a help message. The below example demonstrates the use of the utility, where all arguments except `--out_dir` are required.

```shell
python3 gpdatagen.py \
/odom /taurob_tracker/imu/data /taurob_tracker/cmd_vel_raw \
--label /ground_truth/odom \
--bag bags/taurob_ccw_2022-12-14-16-40-22.bag \
--out_dir ./data
```

where the second line lists the name of all the topics that are used for feature generation and the `--label` flag dictates the single topic that is used to generate labels. Point the encoder to a rosbag file using the `--bag` option and set an output directory where the dataset files can be found vie the `--out_dir` flag.