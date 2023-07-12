from gpmcl.transform import Pose2D, point_to_observation, observation_delta, observation_jacobian
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
            estimated_motion=np.repeat([estimated_motion], repeats=self.M, axis=0),
            previous_motion=self.previous_motion,
        )
        self.previous_motion = predicted_motion
        for idx, particle in enumerate(self.particles):
            # TODO: add noise according to predicted covaraince
            particle.apply_u(predicted_motion[idx])

    def update(self, keypoints: open3d.geometry.PointCloud) -> None:
        """Update the particle poses and landmarks from observed keypoints."""
        # update landmarks
        # compute particle likelihoods
        # normalize weights
        # resmple particles
        Q_0 = np.array(self.config["keypoint_covariance"])
        Q_z = np.array(self.config["observation_covariance"])
        for m, particle in enumerate(self.particles):
            (
                correspondences,
                pcd_keypoints_in_map_frame,
                best_correspondence,
                idxs_new_keypoints,
            ) = particle.estimate_correspondences(keypoints)
            if correspondences.shape[0] == 0:
                # add all keypoints as landmarks
                particle.add_landmarks(
                    ls=np.asarray(pcd_keypoints_in_map_frame.points),
                    Q_0=Q_0,
                )
            else:
                # for updating, we need the keypoints to lie in the robots pose frame
                # see point_to_observation and observation_jacobian!
                keypoints = np.asarray(keypoints.points)
                for correspondence in correspondences:
                    idx_landmark, idx_keypoint = correspondence
                    # update the landamark position at idx_landmark with the keypoint position at idx_keypoint
                    z_i = point_to_observation(keypoints[idx_keypoint])
                    # TODO: this requires the landmarks to be in the robot frame!!
                    z_est_i = particle.landmarks[idx_landmark]
                    delta_z = observation_delta(z_i, z_est_i)
                    Q_i = particle.landmark_covariances[idx_landmark]
                    H_i = observation_jacobian(point=keypoints[idx_keypoint])
                    # TODO: use cholesky-inverse for numerical stability of covariance matrices
                    K_i = Q_i @ H_i.T @ np.linalg.inv(H_i @ Q_i @ H_i.T + Q_z)
                    # update the landmarks location
                    particle.update_landmark(idx=idx_landmark, delta=delta_z, H=H_i, K_gain=K_i, Qz=Q_z)
                # TODO: add previously unseen features, implement feature observation count
                # TODO: compute particle likelihood given the best correspondence
                idx_l_min, idx_z_min = best_correspondence
                self.ws[m] = 1.0  # multivariate PDF here!
        # TODO: compute particle likelihoods, normalize and update particles
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
