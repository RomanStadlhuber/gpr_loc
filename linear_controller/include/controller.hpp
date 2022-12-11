
#include <Eigen/Dense>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#ifndef CONTROLLER_H_
#define CONTROLLER_H_
namespace navigation
{ // a proportional that generate twist efforts from poses
    class Controller
    {
        // current plant state "y"
        geometry_msgs::Pose state;
        // current setpoint "w"
        geometry_msgs::Pose setpoint;
        // norm of difference vector
        double rho;
        // difference vector angle
        double alpha;
        // pose heading difference
        double beta;
        // linear velocity proportionality
        double K_v;
        // relative distance angle proportionality
        double K_alpha;
        // heading difference proportionality
        double K_beta;
        // linear velocity limit
        double v_max;
        // angular velocity limit
        double omega_max;

    public:
        Controller()
        {
            return;
        }
        // initialize controller with controller params and effort limits
        Controller(
            const double &K_v,
            const double &K_alpha,
            const double &K_beta,
            const double &v_max,
            const double &omega_max);
        // set current plant state
        auto set_state(const geometry_msgs::Pose &) -> void;
        // set goal as setpoint for controller
        auto set_goal(const geometry_msgs::Pose &) -> void;
        // get controller effort
        auto get_effort() -> geometry_msgs::Twist;
        // vector of [x,y,theta] to tell the current distance from goal
        auto current_distance() -> Eigen::Vector3d;

    private:
        // compute and buffer control efforts
        auto compute_efforts() -> void;
        // converts individual effort values to a twist (velocty command)
        auto cmd_vel() -> geometry_msgs::Twist;
        // get theta angle from quaternion
        auto pose_theta(const geometry_msgs::Pose&) -> double; 
        // normalize an angle to [-pi ; pi]
        auto normalize_angle(const double &) -> double;
    };
}
#endif