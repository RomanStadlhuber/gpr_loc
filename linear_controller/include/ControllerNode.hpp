#ifndef CONTORLLER_NODE_HPP_
#define CONTORLLER_NODE_HPP_

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <controller.hpp>
#include <vector>
#include <string>

class ControllerNode
{
public:
    ControllerNode(
        const std::string &reference_frame_id,
        const std::string &robot_frame_id,
        const std::string &goal_frame_id);

    // defines callback handler for goal pose
    auto goal_callback(const geometry_msgs::Pose::ConstPtr &goal) -> void;
    auto odom_callback(const nav_msgs::Odometry::ConstPtr &odom) -> void;

protected:
    ros::NodeHandle nh;
    ros::Subscriber goal_sub;
    ros::Subscriber odom_sub;
    ros::Publisher cmd_vel_pub;
    ros::Publisher goal_pub;
    navigation::Controller controller;
    std::vector<geometry_msgs::Pose> trajectory;
    const std::string goal_frame_id, robot_frame_id, reference_frame_id;
    int current_waypoint_idx;
    double K_v;
    double K_alpha;
    double K_beta;
    double v_max;     // [m/s]
    double omega_max; // [rad/s]
    double threshold; // minimum required distancje to goal point
    tf2_ros::TransformBroadcaster tf_broadcaster;

private:
    // load waypoint from ROS param file
    auto load_trajectory() -> void;
    // load controller settings from ROS param file
    auto load_controller_params() -> void;
    // publishes a transform of the pose to the corresponding frame
    auto publish_pose_tf(const std::string &frame_id, const geometry_msgs::Pose &pose) -> void;
};

#endif