import numpy as np
import open3d


class FastSLAM:
    # TODO: how to manage particle values?

    def __init__(self) -> None:
        # initialize particles by setting landmarks empty
        # and sampling about intial guess with uniform distribution
        # load/set inference function to gaussian process models
        pass

    def predict(self, u: np.ndarray) -> None:
        # NOTE: "u" is the motion delta in the robot frame
        # predict new particle poses
        # update the trajectories by appending the prior
        pass

    def update(self, z: open3d.geometry.PointCloud) -> None:
        # detect keypoints
        # update landmarks
        # compute particle likelihoods
        # resmple particles
        pass
