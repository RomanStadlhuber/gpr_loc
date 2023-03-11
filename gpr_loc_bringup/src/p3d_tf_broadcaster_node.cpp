#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

/**
 * NOTE: the following code was adapted from official ROS Wiki Tutorials
 * - http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber
 * - http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20broadcaster
 */

void callback_p3d_groundtruth(const nav_msgs::Odometry::ConstPtr &msg)
{
    static tf::TransformBroadcaster tf_broadcaster;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z));
    transform.setRotation(tf::Quaternion(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w));
    // TODO: make the parent and child frame names parameters!
    tf_broadcaster.sendTransform(
        tf::StampedTransform(transform,
                             ros::Time::now(),
                             "odom",
                             "base_footprint"));
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "p3d_tf_broadcaster");
    ROS_INFO_ONCE("Initiating P3D TF Broadcaster Node.");
    auto nh = ros::NodeHandle("~");
    ros::Subscriber sub = nh.subscribe("ground_truth/odom", 100, callback_p3d_groundtruth);
    while (ros::ok())
    {
        ros::spinOnce();
    }
    ROS_WARN_ONCE("Shutting down P3D TF Broadcaster Node!");
    return 0;
}