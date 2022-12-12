#include <controller.hpp>
#include <math.h>
#define _USE_MATH_DEFINES // required to use pi

namespace navigation
{

    Controller::Controller(
        const double &K_v,      // velocity gain
        const double &K_alpha,  // polar coordinate radius angle gain
        const double &K_beta,   // robot pose angle gain
        const double &v_max,    // speed limit
        const double &omega_max // turn limit
    )
    {
        this->K_v = K_v;
        this->K_alpha = K_alpha;
        this->K_beta = K_beta;
        auto zero_pose = geometry_msgs::Pose();
        this->state = zero_pose;
        this->setpoint = zero_pose;
        this->v_max = v_max;
        this->omega_max = omega_max;
        this->ignore_theta = false;
    }

    auto Controller::set_state(const geometry_msgs::Pose &state) -> void
    {
        this->state = state;
    }

    auto Controller::set_goal(const geometry_msgs::Pose &goal) -> void
    {
        this->setpoint = goal;
    }

    auto Controller::compute_efforts() -> void
    {
        auto const delta_rho = Eigen::Vector2d(
            setpoint.position.x - state.position.x,
            setpoint.position.y - state.position.y);
        this->rho = delta_rho.norm();

        auto const theta_state = pose_theta(state);
        auto const theta_setpoint = pose_theta(
            this->ignore_theta // should the setpoint heading be ignored?
                ? state        // if so, then the current heading is the setpoint heading
                : setpoint     // otherwise, use the setpoint heading
        );

        auto const alpha = std::atan2(delta_rho.y(), delta_rho.x()) - theta_state;
        auto const beta = theta_setpoint - std::atan2(delta_rho.y(), delta_rho.x());

        this->alpha = normalize_angle(alpha);
        this->beta = normalize_angle(beta);
    }

    auto Controller::current_distance() -> Eigen::Vector3d
    {
        return Eigen::Vector3d(
            setpoint.position.x - state.position.x,
            setpoint.position.y - state.position.y,
            pose_theta(setpoint) - pose_theta(state));
    }

    auto Controller::pose_theta(const geometry_msgs::Pose &pose) -> double
    {
        auto const q = tf2::Quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w);

        auto const R = tf2::Matrix3x3(q);
        double roll, pitch, yaw;
        R.getRPY(roll, pitch, yaw);

        return normalize_angle(yaw);
    }

    auto Controller::get_effort() -> geometry_msgs::Twist
    {
        compute_efforts();
        return cmd_vel();
    }

    auto Controller::cmd_vel() -> geometry_msgs::Twist
    {
        // construct the control command
        auto cmd_vel = geometry_msgs::Twist();

        auto const v = this->K_v * this->rho;
        auto const omega = this->K_alpha * this->alpha + this->K_beta * this->beta;

        cmd_vel.linear.x = std::min(v, v_max);
        cmd_vel.angular.z = std::max(std::min(omega, omega_max), -1.0 * omega_max);
        return cmd_vel;
    }

    auto Controller::normalize_angle(const double &theta) -> double
    {
        if (std::abs(theta) <= M_PI)
            return theta;

        auto theta_comp = std::fmod(theta, 2 * M_PI);
        if (theta_comp > M_PI)
            theta_comp = theta_comp - 2 * M_PI;
        else if (theta_comp < -M_PI)
            theta_comp = theta_comp + 2 * M_PI;

        return theta_comp;
    }

    auto Controller::set_ignore_theta(const bool &ignore_theta) -> void
    {
        this->ignore_theta = ignore_theta;
    }
}