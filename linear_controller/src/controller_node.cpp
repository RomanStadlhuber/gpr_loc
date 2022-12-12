#include <ControllerNode.hpp>

// the frames to use when performing control of the agent
// NOTE: they should ideally be independent of any other existing frames
// these frames should only be modified by the controller and no other nodes
#define REFERENCE_FRAME_ID "odom"
#define ROBOT_FRAME_ID "base_link"
#define GOAL_FRAME_ID "control_waypoint"

auto main(int argc, char **argv) -> int
{
    ros::init(argc, argv, "linear_controller");

    auto const controller_node = ControllerNode(
        REFERENCE_FRAME_ID, ROBOT_FRAME_ID, GOAL_FRAME_ID);

    // publish the global reference frame for RVIZ
    // code adapted from:
    // http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20static%20broadcaster%20%28C%2B%2B%29
    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    geometry_msgs::TransformStamped static_transformStamped;
    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = std::string(REFERENCE_FRAME_ID);
    static_transformStamped.child_frame_id = std::string(ROBOT_FRAME_ID);
    static_transformStamped.transform.translation.x = 0;
    static_transformStamped.transform.translation.y = 0;
    static_transformStamped.transform.translation.z = 0;
    tf2::Quaternion quat;
    quat.setRPY(0, 0, 0);
    static_transformStamped.transform.rotation.x = quat.x();
    static_transformStamped.transform.rotation.y = quat.y();
    static_transformStamped.transform.rotation.z = quat.z();
    static_transformStamped.transform.rotation.w = quat.w();
    static_broadcaster.sendTransform(static_transformStamped);
    while (ros::ok())
    {
        ros::spinOnce();
    }
    ROS_WARN_ONCE("shutting down controller node!");
    ros::shutdown();
    return 0;
}