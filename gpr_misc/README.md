# Development Package for GPR-Localization

This package includes the code that is used to develop the gaussian process regression scheme for mobile robot localization.

## Generating Datasets from Rosbags

The following will give an overview of how to generate and interpret GP datasets from resbags.

GP datasets come in the form `D = (X, y)`, where `X` is a matrix of row vectors that represent the input features for the GP and `y` is the row-vector of scalar-valued output labels.

The `rosbag_parser.py` script allows to generate GP datasets in the form of indexed and column-named CSV files (that can be processed using the Pandas library) based on a singel rosbag input.

### Creating Feature Encoders

The parser file uses feature encoding functions that convert ROS messages into feature vectors, with each feature (see `GPFeature` in `helper_types.py`) having both a `column_name` for the exported dataframe and a scalar `value`.

The `rosbags` package provides typings for deserialized ROS messages. In order to create a feature encoder for a specific message type, import the necessary type

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

Simpy run the `rosbag_parser.py` by supplying to it a valid path of a rosbag file and a list of topics for both the features and the labels.