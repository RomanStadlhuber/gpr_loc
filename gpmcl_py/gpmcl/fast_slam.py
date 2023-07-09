from gpmcl.transform import Pose2D
from gpmcl.config import FastSLAMConfig
from gpmcl.particle import FastSLAMParticle
from typing import List
import numpy as np
import scipy.stats
import open3d


class FastSLAM:
    # TODO: how to manage particle values?

    def __init__(self, config: FastSLAMConfig) -> None:
        # initialize particles by setting landmarks empty
        # and sampling about intial guess with uniform distribution
        # load/set inference function to gaussian process models
        self.config = config
        self.M = self.config["particle_count"]
        self.particles: List[FastSLAMParticle] = []
        pass

    def initialize_from_pose(self, x0: np.ndarray) -> None:
        """Initialize the particle set by sampling about a pose."""
        # TODO: configure initial sampling radius
        self.particles = self.sample_circular_uniform(initial_guess=x0)

    def predict(self, u: np.ndarray) -> None:
        # NOTE: "u" is the motion delta in the robot frame
        # predict new particle poses
        # update the trajectories by appending the prior
        pass

    def update(self, keypoints: open3d.geometry.PointCloud) -> None:
        """Update the particle poses and landmarks from observed keypoints."""
        # detect keypoints
        # update landmarks
        # compute particle likelihoods
        # resmple particles
        pass

    def sample_circular_uniform(self, initial_guess: np.ndarray, circle_radius: float = 1.0) -> List[FastSLAMParticle]:
        """Sample **new** particles about an initial guess pose.

        This **resets** the entire particle and map state.
        """
        ranges = scipy.stats.uniform.rvs(scale=circle_radius, size=self.M)
        bearings = scipy.stats.uniform.rvs(loc=-np.pi, scale=2 * np.pi, size=self.M)
        headings = scipy.stats.uniform.rvs(loc=-np.pi, scale=2 * np.pi, size=self.M)
        x0, y0, h0 = initial_guess
        # convert range and bearing values to 2D poses
        poses = list(
            map(
                lambda r, b, h: np.array(r * np.cos(b) + x0, r * np.sin(b) + y0, h + h0),
                zip(ranges, bearings, headings),
            )
        )
        particles = list(map(lambda vec: FastSLAMParticle(x=Pose2D.from_twist(vec)), poses))
        return particles
