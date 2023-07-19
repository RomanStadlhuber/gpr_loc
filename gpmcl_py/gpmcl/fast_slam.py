from gpmcl.transform import Pose2D
from gpmcl.config import FastSLAMConfig
from gpmcl.particle import FastSLAMParticle
from gpmcl.motion_model import MotionModel
from gpmcl.observation_model import ObservationModel
from filterpy.monte_carlo import systematic_resample
from typing import List, Tuple
import numpy as np
import scipy.stats
import open3d


class FastSLAM:
    def __init__(self, config: FastSLAMConfig, motion_model: MotionModel, observation_model: ObservationModel) -> None:
        # initialize particles by setting landmarks empty
        # and sampling about intial guess with uniform distribution
        # load/set inference function to gaussian process models
        self.config = config
        self.M = self.config["particle_count"]
        self.particles: List[FastSLAMParticle] = []
        # initialize all particle weights equally likely
        self.ws = 1 / self.M * np.ones(self.M)
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.previous_motion = np.zeros((self.M, 3), dtype=np.float64)
        pass

    def initialize_from_pose(self, x0: np.ndarray) -> None:
        """Initialize the particle set by sampling about a pose."""
        # region: ciruclar uniform sampling about initial guess
        # TODO: configure initial sampling radius
        # NOTE: the problem with sampling the initial from a distribution is that
        # every pose and heading is equally likely at that point
        # this means that while the trajectory might be accurate,
        # the direction its headed in could be completely off
        # self.particles, self.ws = self.sample_circular_uniform(initial_guess=x0)
        # endregion
        self.particles, self.ws = self.sample_identical(initial_guess=x0)

    def predict(self, estimated_motion: np.ndarray) -> None:
        # predict new particle poses
        # update the trajectories by appending the prior
        predicted_motion, predicted_motion_variances = self.motion_model.predict(
            estimated_motion=np.repeat([estimated_motion], repeats=self.M, axis=0),
            previous_motion=self.previous_motion,
        )
        self.previous_motion = predicted_motion
        for idx, particle in enumerate(self.particles):
            cov = np.diag(predicted_motion_variances[idx])
            particle.apply_u(u=predicted_motion[idx], R=cov)

    def update(self, pcd_keypoints: open3d.geometry.PointCloud) -> float:
        """Update the particle poses and landmarks from observed keypoints.

        Returns the so-called *effective weight* or the relative amount of "useful"
        particles w.r.t. the particle-count.
        """
        # initial landmark covariance
        Q_0 = np.array(self.config["keypoint_covariance"]).reshape((3, 3))
        # range-bearing observation covariance
        Q_z = np.array(self.config["observation_covariance"]).reshape((3, 3))
        keypoints = np.asarray(pcd_keypoints.points)
        # compute distance to all keypoints, used to select only those that are in range
        keypoint_distances = np.linalg.norm(keypoints, axis=1)
        # select only keypoints that lie within the max. feature range
        selected_keypoints = keypoints[np.where(keypoint_distances <= self.config["max_feature_range"])]
        # update the keypoint pcd if there are fewer points in range
        if selected_keypoints.shape != keypoints.shape:
            pcd_keypoints = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(selected_keypoints))
        # update the particle states and their corresponding likelihoods
        for m, particle in enumerate(self.particles):
            (
                correspondences,
                best_correspondence,
                idxs_new_keypoints,
            ) = particle.estimate_correspondences(pcd_keypoints)
            if correspondences.shape[0] == 0:
                # remove the particle (i.e. likelihood to zero) if it has landmarks but no matches
                if particle.landmarks.shape[0] == self.config["max_active_landmarks"]:
                    self.ws[m] = 0
                    continue
                particle.add_new_landmarks_from_keypoints(
                    idxs_new_landmarks=idxs_new_keypoints,
                    keypoints_in_robot_frame=selected_keypoints,
                    position_covariance=Q_0,
                    max_active_landmarks=self.config["max_active_landmarks"],
                )
            else:
                innovations, innovation_covariances = particle.update_existing_landmarks(
                    correspondences=correspondences,
                    keypoints_in_robot_frame=selected_keypoints,
                    observation_covariance=Q_z,
                )
                particle.add_new_landmarks_from_keypoints(
                    idxs_new_landmarks=idxs_new_keypoints,
                    keypoints_in_robot_frame=selected_keypoints,
                    position_covariance=Q_0,
                    max_active_landmarks=self.config["max_active_landmarks"],
                )
                # remove landmarks that are out of range or unobserved too often
                particle.prune_landmarks(
                    max_distance=self.config["max_feature_range"],
                    max_unobserved_count=self.config["max_unobserved_count"],
                )
                # TODO: register the unobserved landmarks (or should it be done internally?)
                idx_l_min, _ = best_correspondence
                # compute a particles likelihood given its best correspondence
                likelihood = self.observation_model.compute_likelihood(
                    dz=innovations[idx_l_min], Q=innovation_covariances[idx_l_min]
                )
                self.ws[m] = likelihood
        # normalize the likelihoods to obtain a nonparametric PDF
        self.ws /= np.sum(self.ws) + 1e-12
        # compute ratio of effective particles
        w_eff = self.compute_effective_weight()
        # compute the indices to resample from
        idxs_resample = systematic_resample(weights=self.ws)
        # create an intermediate set of resampled particles
        intermediate_particles: List[FastSLAMParticle] = []
        for idx_resampled in idxs_resample:
            intermediate_particles.append(FastSLAMParticle.copy(self.particles[idx_resampled]))
        # update the current particles and reset their weights
        self.particles = intermediate_particles
        self.ws = 1 / self.M * np.ones(self.M, dtype=np.float64)
        return w_eff

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

    def sample_identical(self, initial_guess: np.ndarray) -> Tuple[List[FastSLAMParticle], np.ndarray]:
        """Create `self.M` particles at the exact same pose."""
        particles: List[FastSLAMParticle] = []
        for _ in range(self.M):
            particles.append(FastSLAMParticle(x=Pose2D.from_twist(initial_guess)))
        ws = 1 / self.M * np.ones(self.M)
        return (particles, ws)

    def compute_effective_weight(self) -> float:
        """Compute the normalized weights of the effective particles.

        Roughly speaking, returns the ratio `useful_particles / particle_count`.

        See `Eq. (16)` in [(Elfring, Torta, v.d. Molengraft, 2021)](https://www.mdpi.com/1424-8220/21/2/438).
        """
        return 1 / np.sum(np.square(self.ws)) / self.M
