from rosbags.typesys.types import nav_msgs__msg__Odometry as Odometry
from filterpy.monte_carlo import systematic_resample
from transform import Pose2D, point_to_observation, observation_delta
from gpmcl.regression import GPRegression
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
import scipy.stats
import open3d


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
        process_regressor: GPRegression,
        initial_ground_truth: Optional[Odometry] = None,
        initial_odom_estimate: Optional[Odometry] = None,
    ) -> None:
        # the posterior is the initial guess pose
        # self.posterior_pose = Pose2D.from_odometry(initial_ground_truth) if initial_ground_truth else Pose2D()
        # the last odometry estimate is used to compute deltas
        self.odom_last = Pose2D.from_odometry(initial_odom_estimate) if initial_odom_estimate else Pose2D()
        self.posterior_pose = self.odom_last
        # initialize the hyperparameters
        self.M = config.particle_count  # number of particles to track
        self.R = config.process_covariance_R  # process covariance
        self.Q = config.observation_covariance_Q  # observation covariance
        # the distributions used to sample signal & control noise
        self.N_u = scipy.stats.multivariate_normal(mean=np.zeros(3), cov=self.R)  # control noise
        self.N_dx = scipy.stats.multivariate_normal(mean=np.zeros(3), cov=0.065 * np.eye(3))  # last delta noise
        # sample the initial states
        # an N x 3 array representing the particles as twists
        self.Xs = self.__sample_multivariate_normal(self.posterior_pose, 0.1 * np.eye(3))
        self.ws = (1 / self.M) * np.ones(self.M, dtype=np.float64)
        # the last motion delta, used as part of the input to the GP
        noise_dx = self.N_dx.rvs(size=self.M)
        self.dX_last = np.zeros((self.M, 3), dtype=np.float64) + noise_dx
        # the particle weights
        self.M_eff = self.M
        # the gaussian process of the motion model
        self.GP_p = process_regressor

    def predict(self, odom: Odometry) -> None:
        X_est = Pose2D.from_odometry(odom)
        # estimated delta transformation "dU"
        T_delta_u = self.odom_last.inv() @ X_est.T
        delta_u = Pose2D(T_delta_u).as_twist()
        # repeat the estimated motion and add gaussian white noise to spread the particles
        noise_u = self.N_u.rvs(size=self.M)
        noise_dx = self.N_dx.rvs(size=self.M)
        U = np.repeat([delta_u], self.M, axis=0) + noise_u
        self.odom_last = X_est
        # predict the next states from GP regression of the process model
        X_predicted, dX = self.GP_p.predict(self.Xs, dX_last=self.dX_last, U=U)
        # update both the particles and their last state changes
        self.dX_last = dX + noise_dx
        self.Xs = X_predicted
        # set posterior (as prior)
        self.posterior_pose = self.mean()

    def update(
        self,
        ground_truth: Optional[Odometry] = None,
        features_and_landmarks: Optional[Tuple[open3d.geometry.PointCloud, open3d.geometry.PointCloud]] = None,
    ) -> None:
        """Update the particle states using observed landmarks.

        ### Parameters
        `Z` - The observed features.
        `mapper` - An instance of `gpmcl.mapper.Mapper` used for updating particle states.
        """
        # update the particles if groundtruth is provided
        if ground_truth is not None:
            # x and y position acting as GPS signal
            pos_gps = Pose2D.from_odometry(ground_truth).as_twist()[:2]
            N_pgs = scipy.stats.multivariate_normal(mean=pos_gps, cov=self.Q[:2, :2])
            # measurements are the current particle filter positions
            z = self.Xs[:, :2]
            qs = N_pgs.pdf(z)
            self.ws = qs
            # normalize all weights
            self.ws /= np.sum(self.ws) + 1e-30

        if features_and_landmarks is not None:
            pcd_features, pcd_landmarks = features_and_landmarks
            # feature observations are the same for every particle
            for idx, x in enumerate(self.Xs):
                # 3d affine transform representing the inverse of the particles pose
                T_x_i_inv = Pose2D.from_twist(x).inv3d()
                # copy of the landmark-pcd for use in this particles frame
                # a copy is required because in-place transform would otherwise act on the original data
                pcd_landmarks_in_x_i = open3d.geometry.PointCloud(pcd_landmarks)
                # transform the landmarks into the particles pose frame
                pcd_landmarks_in_x_i.transform(T_x_i_inv)
                corresponding_points = self.__pick_closest_correspondences(pcd_features, pcd_landmarks_in_x_i, 5)
                corresponding_observations = list(
                    map(
                        lambda pt_tuple: (point_to_observation(pt_tuple[0]), point_to_observation(pt_tuple[1])),
                        corresponding_points,
                    )
                )
                # compute the error of two corresponding observations
                observation_errors_i = list(
                    map(lambda obs: observation_delta(obs[0], obs[1]), corresponding_observations)
                )
                # compute the likelihood of i-th particle given its observation errors
                self.ws[idx] = self.__observation_likelihood(deltas=observation_errors_i)
            # normalize all weights
            self.ws /= np.sum(self.ws) + 1e-30

        # re-initialize particles if the effective is too low
        self.M_eff = 1 / np.sum(np.square(self.ws))
        if self.M_eff <= self.M / 2:
            # reset particles and weights
            self.Xs = self.__sample_multivariate_normal(
                Pose2D.from_odometry(ground_truth) if ground_truth is not None else self.mean(),
                0.1 * np.eye(3),  # sample covariance
            )
            self.ws = (1 / self.M) * np.ones(self.M, dtype=np.float64)
            # reset the last motion deltas
            noise_dx = self.N_dx.rvs(size=self.M)
            self.dX_last = np.zeros((self.M, 3), dtype=np.float64) + noise_dx
            return

        self.__resample()

    def mean(self) -> Pose2D:
        """Compute the mean state as pose object."""
        # get the mean particle state as twist
        x_mu = self.__compute_mean_from_sample_points()
        # convert to pose object and return
        return Pose2D.from_twist(x_mu)

    def __sample_multivariate_normal(self, X0: Pose2D, S: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample particles about the desired state.

        Samples are drawn with replacement from a multivariate normal using the process
        covariance `R`.

        Resets all particles to equal weights. Resets `self.dX_last` to all zeros.
        """
        x0 = X0.as_twist()
        cov = S if S is not None else self.R
        # the particle statesj
        Xs = np.random.default_rng().multivariate_normal(x0, cov, (self.M,))
        return Xs

    def __compute_mean_from_sample_points(self) -> np.ndarray:
        """Compute the posterior"""
        return np.average(self.Xs, weights=self.ws, axis=0)

    def __resample(self) -> None:
        """Resample the particles representing the state.

        Is internally setting
        - the particle states `self.Xs`
        - the weights `self.ws`
        - the last motion deltas `self.dX_last`

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
        self.dX_last = dX_last_resampled

    def __pick_closest_correspondences(
        self,
        pcd_features: open3d.geometry.PointCloud,
        pcd_landmarks: open3d.geometry.PointCloud,
        pick_n_closest: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        pts_f = np.asarray(pcd_features.points)
        pts_l = np.asarray(pcd_landmarks.points)
        distances = np.linalg.norm((pts_l - pts_f), axis=1)
        # indices that sort the points according to distances in ascending order
        idxs_sorted_asc = np.argsort(distances)
        pts_f = pts_f[idxs_sorted_asc]
        pts_l = pts_l[idxs_sorted_asc]
        return list(
            map(
                lambda i: (pts_f[i], pts_l[i]),
                range(
                    np.min(
                        [
                            pick_n_closest,
                            len(idxs_sorted_asc),
                        ]
                    )
                ),
            )
        )

    def __observation_likelihood(self, deltas: List[np.ndarray]) -> float:
        # TODO: implement probability computation (either from GP or normal distributions)
        # convert the list to a matrix for bulk-processing
        delta_mat = np.array(deltas)
        # construct a diagonal covariance matrix from the independent observation axes
        cov = np.diag([0.1, 0.2, 0.2])
        # compute likeihood of all observations
        ps = scipy.stats.multivariate_normal.pdf(x=delta_mat, mean=np.zeros(3), cov=cov, allow_singular=False)
        # total likelihood is the product of all individual observation likelihoods
        p = np.product(ps)
        return p
