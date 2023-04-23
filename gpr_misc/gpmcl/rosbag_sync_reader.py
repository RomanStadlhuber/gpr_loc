from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from typing import Set, Dict, Optional, Callable
from abc import ABC, abstractmethod
import pathlib


class SyncMessage(ABC):
    """Abstract base class for messages containing synchronized data"""

    # NOTE: this is just used for mypy
    # using @abstractstaticmethod would be correct as well
    @staticmethod
    @abstractmethod
    def from_dict(d: Dict) -> Optional["SyncMessage"]:
        """Obtain a class instance from a dictionary."""
        pass


class RosbagSyncReader:
    """A utility class to obtain synchronized data"""

    def __init__(self, bag_path: pathlib.Path) -> None:
        self.bag_path = bag_path

    # TODO: make the callback argument a generic inherited from synced object
    def spin(
        self,
        topics: Set,
        callback: Callable[[Optional[Dict], Optional[int]], None],
        grace_period_secs: float = 1e-10,
    ):
        """Synchronize a set of topics.

        Optionally provides a grace period in seconds.
        """

        if not self.bag_path.exists() or not self.bag_path.is_file():
            raise FileNotFoundError(f"{self.bag_path} doesn't exist or is not a rosbag.")

        try:
            with Reader(self.bag_path) as reader:
                # convert grace period to nsecs (used for timestamp comparsion)
                grace_period_nsecs = int(grace_period_secs * 10e9)
                # a buffer used to store messages during synchronization
                buffered_messages: Dict = dict()
                # the last sync timestamp
                sync_start = 0

                def has_all_messages() -> bool:
                    return set(buffered_messages.keys()) == topics

                def has_no_messages() -> bool:
                    return len(buffered_messages.keys()) == 0

                for connection, timestamp, rawdata in reader.messages():
                    # skip if topic not desired
                    if connection.topic not in topics:
                        continue

                    if has_no_messages():
                        sync_start = timestamp
                        msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                        buffered_messages[connection.topic] = msg
                    else:
                        time_delta = timestamp - sync_start
                        # reset syncer if grace period elapsed
                        if time_delta > grace_period_nsecs:
                            buffered_messages.clear()
                            # empty callback because sync failed
                            callback(None, None)
                        # buffer messages if still below grace period
                        else:
                            msg = deserialize_cdr(
                                ros1_to_cdr(rawdata, connection.msgtype),
                                connection.msgtype,
                            )
                            buffered_messages[connection.topic] = msg
                            # callback with synced data if sync was successful
                            if has_all_messages():
                                callback(buffered_messages, timestamp)
                                buffered_messages.clear()

        except Exception as e:
            print(str(e))
