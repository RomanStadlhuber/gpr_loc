#include <ControllerNode.hpp>

ControllerNode::ControllerNode(
    const std::string &reference_frame_id,
    const std::string &robot_frame_id,
    const std::string &goal_frame_id

    ) : reference_frame_id(reference_frame_id),
        robot_frame_id(robot_frame_id),
        goal_frame_id(goal_frame_id)
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

    controller.set_ignore_theta(ignore_yaw);
    // set controller inactive until the first goal arrives
    controller.set_is_active(false);
}

// defines callback handler for goal pose
auto ControllerNode::goal_callback(const geometry_msgs::Pose::ConstPtr &goal) -> void
{
    controller.set_goal(*goal);

    // set controller active as soon as a goal comes in
    if (!controller.get_is_active())
        controller.set_is_active(true);

    auto const cmd_vel = controller.get_effort();
    cmd_vel_pub.publish(cmd_vel);
    publish_pose_tf(std::string(goal_frame_id), *goal);
}
// defines callback for current pose
auto ControllerNode::odom_callback(const nav_msgs::Odometry::ConstPtr &odom) -> void
{
    controller.set_state(odom->pose.pose);

    // publish current waypoint
    goal_pub.publish(trajectory[current_waypoint_idx]);
    publish_pose_tf(std::string(robot_frame_id), odom->pose.pose);

    if (!controller.get_is_active())
        return;

    auto const cmd_vel = controller.get_effort();
    cmd_vel_pub.publish(cmd_vel);
    // update goal waypoint if controller is close enough
    auto const delta = controller.current_distance();
    // TODO: check yaw if controller doesn't ignore it
    auto const close_enough =
        Eigen::Vector2d(delta.x(), delta.y()).norm() <= threshold;
    if (close_enough && controller.get_is_active())
    {
        current_waypoint_idx++;
        if (current_waypoint_idx > trajectory.size() - 1)
            current_waypoint_idx = 0;
        ROS_INFO("close enough, updating goal");
        // after goal update, a re-publish is needed
        // otherwise the controller does not receive the latest goal
        // this would lead it to skip wayoints for all odom updates
        // that occur before the next goal update
        goal_pub.publish(trajectory[current_waypoint_idx]);
        publish_pose_tf(std::string(robot_frame_id), odom->pose.pose);
    }
}

// load waypoint from ROS param file
auto ControllerNode::load_trajectory() -> void
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
auto ControllerNode::load_controller_params() -> void
{
    XmlRpc::XmlRpcValue controller_parameters;
    if (!nh.getParam("controller", controller_parameters))
    {
        ROS_ERROR_ONCE("No controller parameters were provided!");
        ROS_WARN_ONCE("initiating shutdown..");
        ros::shutdown();
    }
    else
    {
        // controller parameters
        const double K_v = controller_parameters["K_v"];
        const double K_alpha = controller_parameters["K_alpha"];
        const double K_beta = controller_parameters["K_beta"];
        // limits
        const double v_max = controller_parameters["v_max"];
        const double omega_max = controller_parameters["omega_max"];

        // goal success threshold
        const double threshold = controller_parameters["threshold"];

        const bool ignore_yaw = controller_parameters["ignore_yaw"];

        this->K_v = K_v;
        this->K_alpha = K_alpha;
        this->K_beta = K_beta;
        this->v_max = v_max;
        this->omega_max = omega_max;
        this->ignore_yaw = ignore_yaw;
        this->threshold = threshold;
    }
}

// publishes a transform of the pose to the corresponding frame
auto ControllerNode::publish_pose_tf(const std::string &frame_id, const geometry_msgs::Pose &pose) -> void
{
    // code adapted from
    // http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20broadcaster%20%28C%2B%2B%29
    auto tf = geometry_msgs::TransformStamped();
    // set metadata
    tf.header.frame_id = std::string(reference_frame_id);
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