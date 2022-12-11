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

#define REFERENCE_FRAME_ID "map"
#define ROBOT_FRAME_ID "robot"
#define GOAL_FRAME_ID "goal"

// class orchestrating the controller and the published trajectories
class ControllerNode
{
public:
    ControllerNode()
    {
        nh = ros::NodeHandle("~");
        goal_sub = nh.subscribe("goal", 1, &ControllerNode::goal_callback, this);
        odom_sub = nh.subscribe("odom", 1, &ControllerNode::odom_callback, this);
        cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1);
        goal_pub = nh.advertise<geometry_msgs::Pose>("goal", 1);

        this->load_controller_params();
        this->load_trajectory();

        controller = navigation::Controller(
            K_v, K_alpha, K_beta, v_max, omega_max);
    };

    // defines callback handler for goal pose
    auto goal_callback(const geometry_msgs::Pose::ConstPtr &goal) -> void
    {
        controller.set_goal(*goal);
        auto const cmd_vel = controller.get_effort();
        cmd_vel_pub.publish(cmd_vel);
        publish_pose_tf(std::string(GOAL_FRAME_ID), *goal);
    };
    // defines callback for current pose
    auto odom_callback(const nav_msgs::Odometry::ConstPtr &odom) -> void
    {
        controller.set_state(odom->pose.pose);
        auto const cmd_vel = controller.get_effort();
        cmd_vel_pub.publish(cmd_vel);
        // update goal waypoint if controller is close enough
        auto const delta = controller.current_distance();
        auto const close_enougth = 
            Eigen::Vector2d(delta.x(), delta.y()).norm() <= threshold
            && delta.z() <= threshold;
        if (close_enougth)
        {
            current_waypoint_idx++;
            if (current_waypoint_idx > trajectory.size()-1)
                current_waypoint_idx = 0;
            ROS_INFO("close enough, updating goal");
        }
        // publish current waypoint
        goal_pub.publish(trajectory[current_waypoint_idx]);
        publish_pose_tf(std::string(ROBOT_FRAME_ID), odom->pose.pose);
        publish_pose_tf(std::string("base_link"), odom->pose.pose);
        publish_pose_tf(std::string("base_footprint"), odom->pose.pose);
        publish_pose_tf(std::string("odom"), odom->pose.pose);

    };

protected:
    ros::NodeHandle nh;
    ros::Subscriber goal_sub;
    ros::Subscriber odom_sub;
    ros::Publisher cmd_vel_pub;
    ros::Publisher goal_pub;
    navigation::Controller controller;
    std::vector<geometry_msgs::Pose> trajectory;
    int current_waypoint_idx;
    double K_v;
    double K_alpha;
    double K_beta;
    double v_max;     // [m/s]
    double omega_max; // [rad/s]
    double threshold; // minimum required distance to goal point
    tf2_ros::TransformBroadcaster tf_broadcaster;

private:
    // load waypoint from ROS param file
    auto load_trajectory() -> void
    {
        // reset waypoint index
        current_waypoint_idx = 0;
        // initialize trajectory vector
        trajectory = std::vector<geometry_msgs::Pose>();
        // buffer for trajectory parameter list
        XmlRpc::XmlRpcValue trajectory_list;
        if (!nh.getParam("trajectory", trajectory_list))
        {
            ROS_ERROR_ONCE("No trajectory points were provided!");
        }
        else
        {
            // go through all items in list and set as parameter
            for (int i = 0; i < trajectory_list.size(); i++)
            {
                // waypoint specified in the parameter list
                auto waypoint = geometry_msgs::Pose();
                waypoint.position.x = trajectory_list[i]["x"];
                waypoint.position.y = trajectory_list[i]["y"];
                auto const theta = trajectory_list[i]["theta"];
                tf2::Quaternion q;
                q.setRPY(0, 0, theta);
                waypoint.orientation.x = q.x();
                waypoint.orientation.y = q.y();
                waypoint.orientation.z = q.z();
                waypoint.orientation.w = q.w();
                // add waypoint to trajectory
                trajectory.push_back(waypoint);
            }
            ROS_INFO_ONCE("Added %d waypoints to trajectory!", trajectory_list.size());
        }
        return;
    }

    // load controller settings from ROS param file
    auto load_controller_params() -> void
    {
        // buffer for controller parameters
        XmlRpc::XmlRpcValue controller_params;
        if (!nh.getParam("controller", controller_params))
        {
            ROS_ERROR_ONCE("No controller parameters were supplied!");
        }
        else
        {
            // controller parameters
            K_v = controller_params["K_v"];
            K_alpha = controller_params["K_alpha"];
            K_beta = controller_params["K_beta"];
            // limits
            v_max = controller_params["v_max"];
            omega_max = controller_params["omega_max"];
            // goal success threshold
            threshold = controller_params["threshold"];
        }
    }

    // publishes a transform of the pose to the corresponding frame
    auto publish_pose_tf(const std::string & frame_id, const geometry_msgs::Pose & pose) -> void{
        // code adapted from
        // http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20broadcaster%20%28C%2B%2B%29
        auto tf = geometry_msgs::TransformStamped();
        // set metadata
        tf.header.frame_id = std::string(REFERENCE_FRAME_ID);
        tf.child_frame_id = std::string(frame_id);
        tf.header.stamp = ros::Time::now(); // NOTE: "seq" is set automatically
        // set position
        tf.transform.translation.x = pose.position.x;
        tf.transform.translation.y = pose.position.y;
        tf.transform.translation.z = pose.position.z;
        // set orientation
        tf.transform.rotation.x = pose.orientation.x;
        tf.transform.rotation.y = pose.orientation.y;
        tf.transform.rotation.z = pose.orientation.z;
        tf.transform.rotation.w = pose.orientation.w;
        // publish the tf
        tf_broadcaster.sendTransform(tf);
    }
};

auto main(int argc, char **argv) -> int
{
    ros::init(argc, argv, "linear_controller");

    // publish the global reference frame for RVIZ
    // code adapted from:
    // http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20static%20broadcaster%20%28C%2B%2B%29
    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    geometry_msgs::TransformStamped static_transformStamped;
    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = std::string(REFERENCE_FRAME_ID);
    static_transformStamped.child_frame_id = std::string("base_link");
    static_transformStamped.transform.translation.x = 0;
    static_transformStamped.transform.translation.y = 0;
    static_transformStamped.transform.translation.z = 0;
    tf2::Quaternion quat;
    quat.setRPY(0,0,0);
    static_transformStamped.transform.rotation.x = quat.x();
    static_transformStamped.transform.rotation.y = quat.y();
    static_transformStamped.transform.rotation.z = quat.z();
    static_transformStamped.transform.rotation.w = quat.w();
    static_broadcaster.sendTransform(static_transformStamped);


    auto const controller_node = ControllerNode();
    while (ros::ok())
    {
        ros::spinOnce();
    }
    ROS_WARN_ONCE("shutting down controller node!");
    ros::shutdown();
    return 0;
}