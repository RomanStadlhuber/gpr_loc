from rosbags.typesys.types import nav_msgs__msg__Odometry as Odometry
from filterpy.monte_carlo import systematic_resample
from transform import Pose2D
from gpmcl.mapper import FeatureMap3D, Mapper
from gpmcl.regression import GPRegression
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class ParticleFilterConfig:
    particle_count: int
    process_covariance_R: np.ndarray
    observation_covariance_Q: np.ndarray

    @staticmethod
    def from_config(config: Dict) -> "ParticleFilterConfig":
        """Load configuration from a `PyYAML.load` config document."""
        pf_conf: Dict = config["particle_filter"]
        return ParticleFilterConfig(
            particle_count=pf_conf["particle_count"],
            process_covariance_R=np.reshape(pf_conf["process_covariance_R"], (3, 3)),
            observation_covariance_Q=np.reshape(pf_conf["observation_covariance_Q"], (3, 3)),
        )


class ParticleFilter:
    def __init__(
        self,
        config: ParticleFilterConfig,
        mapper: Mapper,
        process_regressor: GPRegression,
        initial_ground_truth: Optional[Odometry] = None,
        initial_odom_estimate: Optional[Odometry] = None,
    ) -> None:
        # the posterior is the initial guess pose
        self.posterior_pose = Pose2D.from_odometry(initial_ground_truth) if initial_ground_truth else Pose2D()
        # the last odometry estimate is used to compute deltas
        self.U_last = Pose2D.from_odometry(initial_odom_estimate) if initial_odom_estimate else Pose2D()
        # initialize the hyperparameters
        self.M = config.particle_count  # number of particles to track
        self.R = config.process_covariance_R  # process covariance
        self.Q = config.observation_covariance_Q  # observation covariance
        # sample the initial states
        # an N x 3 array representing the particles as twists
        self.Xs = self.__sample_multivariate_normal(self.posterior_pose)
        # the last pose deltas as twists, required for GP inference
        self.dX_last = np.zeros((self.M, 3), dtype=np.float64)
        # the particle weights
        self.ws = (1 / self.M) * np.ones(self.M, dtype=np.float64)  # normalized
        # the gaussian process of the motion model
        self.GP_p = process_regressor
        # the mapper
        self.mapper = mapper

    def predict(self, U: Odometry) -> None:
        dU = self.__compute_dU(U)
        # predict the next states from GP regression of the process model
        X_predicted, dX = self.GP_p.predict(self.Xs, dX_last=self.dX_last, dU=dU)
        # update both the particles and their last state changes
        self.dX_last = dX
        self.Xs = X_predicted
        # set posterior (as prior)
        self.posterior_pose = self.mean()

    def update(self, Z: FeatureMap3D) -> None:
        """Update the particle states using observed landmarks.

        ### Parameters
        `Z` - The observed features.
        `mapper` - An instance of `gpmcl.mapper.Mapper` used for updating particle states.
        """

        # TODO: should this function directly use the raw PCD instead?
        # otherwise the mapper needs to be accessed externally...

        # skip update if no features or landmarks available
        if len(Z.features) == 0 or len(self.mapper.get_map().features) == 0:
            self.__resample()
            self.posterior_pose = self.mean()
            return

        def get_particle_weight(x: np.ndarray) -> float:
            """Compute log-likelihood for a particle state.

            Uses the currently observed features and the gobal map.
            Comutes log-likelihood either from a Gaussian PDF or a
            Gaussian Process based on the filters configuration.
            """
            predicted_pose = Pose2D.from_twist(x)
            correspondences = self.mapper.correspondence_search(
                observed_features=Z,
                pose=predicted_pose.T,
            )
            # TODO: what if 0 correspondences could be found?
            # TODO: visualize map & feature inliers
            # the likelihoods for all feature-landmark correspondences
            likelihoods = self.mapper.get_observation_likelihoods(
                observed_features=Z,
                pose=predicted_pose.T,
                correspondences=correspondences,
            )
            return likelihoods.sum()

        # update the weights for each particle
        if len(self.mapper.get_map().features) > 0:
            # do not incorporate posterior likelihood (see GP Bayes Filters paper, Table 1)
            self.ws = np.array(list(map(get_particle_weight, self.Xs)))
            # resample the particles if their sum is too low
            # NOTE: this also accounts for no correspondences found
            if np.sum(self.ws) <= 1e-5 or np.any(np.isnan(self.ws)):
                self.ws = 1 / self.M * np.ones(np.shape(self.ws))
                self.__resample()
                self.posterior_pose = self.mean()
                return
            # re-normalize the weights based on the new likelihood sum
            self.ws = 1 / (np.sum(self.ws)) * self.ws
        # compute effective weight (see eq. 18 in https://www.mdpi.com/1424-8220/21/2/438)
        N_eff = 1 / np.sum(np.square(self.ws))
        # resample the particles if the effective weights drops below half of the particles
        # again, see the link to the publication above
        if N_eff < (self.M / 2):
            self.__resample()
        # set posterior
        self.posterior_pose = self.mean()

    def mean(self) -> Pose2D:
        """Compute the mean state as pose object."""
        # get the mean particle state as twist
        x_mu = self.__compute_mean_from_sample_points()
        # convert to pose object and return
        return Pose2D.from_twist(x_mu)

    def __sample_multivariate_normal(self, X0: Pose2D) -> np.ndarray:
        """Sample particles from a multivariate normal.

        Uses the process noise covariance `R`.
        Returns an `N x DIM` array representing the sampled particles.
        """
        x0 = X0.as_twist()
        # the particle states
        Xs = np.random.default_rng().multivariate_normal(x0, self.R, (self.M,))
        return Xs

    def __compute_dU(self, U: Odometry) -> np.ndarray:
        """Computes the estimated motion `dU` which is an input to the process GP."""
        X_est = Pose2D.from_odometry(U)
        # estimated delta transformation "dU"
        T_delta_u = self.U_last.inv() @ X_est.T
        delta_u = Pose2D(T_delta_u).as_twist()
        self.U_last = X_est
        w = np.random.default_rng().multivariate_normal(np.zeros(3), self.R, self.M)
        # repeat the estimated motion and add gaussian white noise to spread the particles
        dU = np.repeat([delta_u], self.M, axis=0) + w
        # return the estimated odome
        return dU

    def __compute_mean_from_sample_points(self) -> np.ndarray:
        """Compute the posterior"""
        # TODO: based on weights and particles, compute the single posterior pose
        return np.average(self.Xs, weights=self.ws, axis=0)

    def __resample(self) -> None:
        """Resample the particles representing the state.

        Uses `filterpy.monte_carlo.systematic_resample`."""
        idxs_resampled = systematic_resample(self.ws)
        # number of partiles resampled (largely a safeguard)
        M, *_ = np.shape(idxs_resampled)
        Xs_resampled = self.Xs[idxs_resampled]  # resample the particle states
        dX_last_resampled = self.dX_last[idxs_resampled]  # resample the last state changes
        # evenly redistribute weights among all particles
        ws_resapmled = np.repeat(1 / M, M)
        self.M = M
        self.Xs = Xs_resampled
        self.ws = ws_resapmled
