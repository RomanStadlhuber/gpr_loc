from gpmcl.transform import Pose2D
from gpmcl.config import FastSLAMConfig
from gpmcl.particle import FastSLAMParticle
from gpmcl.motion_model import MotionModel
from typing import List, Tuple
import numpy as np
import scipy.stats
import open3d


class FastSLAM:
    def __init__(self, config: FastSLAMConfig, motion_model: MotionModel) -> None:
        # initialize particles by setting landmarks empty
        # and sampling about intial guess with uniform distribution
        # load/set inference function to gaussian process models
        self.config = config
        self.M = self.config["particle_count"]
        self.particles: List[FastSLAMParticle] = []
        # initialize all particle weights equally likely
        self.ws = 1 / self.M * np.ones(self.M)
        self.motion_model = motion_model
        self.previous_motion = np.zeros((self.M, 3), dtype=np.float64)
        pass

    def initialize_from_pose(self, x0: np.ndarray) -> None:
        """Initialize the particle set by sampling about a pose."""
        # TODO: configure initial sampling radius
        self.particles, self.ws = self.sample_circular_uniform(initial_guess=x0)

    def predict(self, estimated_motion: np.ndarray) -> None:
        # predict new particle poses
        # update the trajectories by appending the prior
        predicted_motion = self.motion_model.predict(
            estimated_motion=np.repeat(estimated_motion, repeats=self.M, axis=0),
            previous_motion=self.previous_motion,
        )
        self.previous_motion = predicted_motion
        for idx, particle in enumerate(self.particles):
            # TODO: add noise according to predicted covaraince
            particle.apply_u(predicted_motion[idx])

    def update(self, keypoints: open3d.geometry.PointCloud) -> None:
        """Update the particle poses and landmarks from observed keypoints."""
        # detect keypoints
        # update landmarks
        # compute particle likelihoods
        # normalize weights
        # resmple particles
        pass

    def get_mean_pose(self) -> Pose2D:
        """Compute the weighted average pose given all particles and their weights."""
        xs = np.array([p.x.as_twist() for p in self.particles])
        mean_pose_twist = np.average(xs, axis=0, weights=self.ws)
        return Pose2D.from_twist(mean_pose_twist)

    def get_particle_poses(self) -> np.ndarray:
        """Obtain the current poses of all particles as twist vectors."""
        xs = np.array([p.x.as_twist() for p in self.particles])
        return xs

    def sample_circular_uniform(
        self, initial_guess: np.ndarray, circle_radius: float = 1.0
    ) -> Tuple[List[FastSLAMParticle], np.ndarray]:
        """Sample **new** particles about an initial guess pose.

        This **resets** the entire particle and map state.

        Returns particles and their weights
        """
        ranges = scipy.stats.uniform.rvs(scale=circle_radius, size=self.M)
        bearings = scipy.stats.uniform.rvs(loc=-np.pi, scale=2 * np.pi, size=self.M)
        headings = scipy.stats.uniform.rvs(loc=-np.pi, scale=2 * np.pi, size=self.M)
        x0, y0, h0 = initial_guess
        # convert range and bearing values to 2D poses
        poses = list(
            map(
                lambda rbh: np.array([rbh[0] * np.cos(rbh[1]) + x0, rbh[0] * np.sin(rbh[1]) + y0, rbh[2] + h0]),  # type: ignore
                zip(ranges, bearings, headings),
            )
        )
        particles = list(map(lambda vec: FastSLAMParticle(x=Pose2D.from_twist(vec)), poses))
        ws = 1 / self.M * np.ones(self.M)
        return (particles, ws)
