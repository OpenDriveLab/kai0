#include "arx_x5_controller/X5Controller.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <csignal>

// using namespace std::chrono_literals;

namespace arx::x5 {
X5Controller::X5Controller() : Node("x5_controller_node") {
  RCLCPP_INFO(this->get_logger(), "机械臂开始初始化...");
  rclcpp::on_shutdown(std::bind(&X5Controller::cleanup, this));
  std::string arm_control_type = this->declare_parameter("arm_control_type", "normal");
  std::string package_name = "arx_x5_controller";
  std::string package_share_dir = ament_index_cpp::get_package_share_directory(package_name);
  int end_type = this->declare_parameter("arm_end_type", 0);
  catch_control_mode_ = static_cast<CatchControlMode>(this->declare_parameter("catch_control_mode", 0));
  go_home_positions_ = this->declare_parameter<std::vector<double>>("go_home_position",std::vector<double>{0,0,0,0,0,0});
  std::string urdf_path;
  if (end_type == 0)
    urdf_path = package_share_dir + "/" + "x5.urdf";
  else if (end_type == 1)
    urdf_path = package_share_dir + "/" + "x5_master.urdf";
  else {
    urdf_path = package_share_dir + "/x5_2025.urdf";
    new_version_ = true;
  }
  interfaces_ptr_ =
      std::make_shared<InterfacesThread>(urdf_path, this->declare_parameter("arm_can_id", "can0"), end_type);
  // RCLCPP_INFO(this->get_logger(), "arm_control_type = %s",arm_control_type.c_str());
  interfaces_ptr_->arx_x(150, 600, 10);
  interfaces_ptr_->setHomePositions(go_home_positions_);
  auto pub_name = this->declare_parameter("arm_pub_topic_name", "arm_status");

  if (arm_control_type == "normal") {
    RCLCPP_INFO(this->get_logger(), "常规模式启动");
    // 创建发布器
    joint_state_publisher_ = this->create_publisher<arx5_arm_msg::msg::RobotStatus>(pub_name, 1);
    // 创建订阅器
    joint_state_subscriber_ = this->create_subscription<arx5_arm_msg::msg::RobotCmd>(
        this->declare_parameter("arm_sub_topic_name", "arm_cmd"),
        10,
        std::bind(&X5Controller::CmdCallback, this, std::placeholders::_1));
    // 定时器，用于发布关节信息
    timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&X5Controller::PubState, this));
  } else if (arm_control_type == "vr_slave") {
    RCLCPP_INFO(this->get_logger(), "vr遥操作模式启动");
    // 创建发布器
    vr_joint_state_publisher_ = this->create_publisher<arm_control::msg::PosCmd>(pub_name, 10);

    joint_state_publisher_ = this->create_publisher<arx5_arm_msg::msg::RobotStatus>(pub_name + "_full", 1);
    // 创建订阅器
    vr_joint_state_subscriber_ = this->create_subscription<arm_control::msg::PosCmd>(
        this->declare_parameter("arm_sub_topic_name", "ARX_VR_L"),
        10,
        std::bind(&X5Controller::VrCmdCallback, this, std::placeholders::_1));
    // 定时器，用于发布关节信息
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&X5Controller::VrPubState, this));
  } else if (arm_control_type == "remote_master") {
    RCLCPP_INFO(this->get_logger(), "remote主机模式启动");
    joint_state_publisher_ = this->create_publisher<arx5_arm_msg::msg::RobotStatus>(pub_name, 10);
    interfaces_ptr_->setArmStatus(InterfacesThread::state::G_COMPENSATION);
    // 定时器，用于发布关节信息
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&X5Controller::PubState, this));
  } else if (arm_control_type == "remote_slave") {
    RCLCPP_INFO(this->get_logger(), "remote从机模式启动");
    joint_state_publisher_ = this->create_publisher<arx5_arm_msg::msg::RobotStatus>(pub_name, 10);
    follow_joint_state_subscriber_ = this->create_subscription<arx5_arm_msg::msg::RobotStatus>(
        this->declare_parameter("arm_sub_topic_name", "followed_arm_topic"),
        10,
        std::bind(&X5Controller::FollowCmdCallback, this, std::placeholders::_1));
    // 定时器，用于发布关节信息
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&X5Controller::PubState, this));
  }
  else if (arm_control_type == "joint_control_v1")
  {
    RCLCPP_INFO(this->get_logger(),"关节控制模式启动");
    joint_state_publisher_ = this->create_publisher<arx5_arm_msg::msg::RobotStatus>(pub_name, 10);
    joint_control_subscriber_ = this->create_subscription<arm_control::msg::JointControl>(
        this->declare_parameter("arm_sub_topic_name", "joint_control"),
        10,
        std::bind(&X5Controller::JointControlCallback,this,std::placeholders::_1));
    timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&X5Controller::PubState, this));
  }
  arx_joy_sub_ = this->create_subscription<std_msgs::msg::Int32MultiArray>("/arx_joy",1,std::bind(&X5Controller::arxJoyCB,this,std::placeholders::_1));
}

void X5Controller::CmdCallback(const arx5_arm_msg::msg::RobotCmd::SharedPtr msg) {
  double end_pos[6] = {
      msg->end_pos[0], msg->end_pos[1], msg->end_pos[2], msg->end_pos[3], msg->end_pos[4], msg->end_pos[5]};

  Eigen::Isometry3d transform = solve::Xyzrpy2Isometry(end_pos);

  interfaces_ptr_->setEndPose(transform);

  std::vector<double> joint_positions = {0, 0, 0, 0, 0, 0};

  for (int i = 0; i < 6; i++) {
    joint_positions[i] = msg->joint_pos[i];
  }

  interfaces_ptr_->setJointPositions(joint_positions);

  interfaces_ptr_->setArmStatus(msg->mode);

  interfaces_ptr_->setCatch(msg->gripper);
}

void X5Controller::arxJoyCB(const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
  if (msg->data[0] == 1)
    interfaces_ptr_->setArmStatus(InterfacesThread::state::G_COMPENSATION);
  if (msg->data[1] == 1)
    interfaces_ptr_->setArmStatus(InterfacesThread::state::GO_HOME);
}

void X5Controller::PubState() {
  auto message = arx5_arm_msg::msg::RobotStatus();
  message.header.stamp = this->get_clock()->now();

  Eigen::Isometry3d transform = interfaces_ptr_->getEndPose();

  // 创建长度为6的vector
  std::array<double, 6> result;

  std::vector<double> xyzrpy = {0, 0, 0, 0, 0, 0};
  xyzrpy = solve::Isometry2Xyzrpy(transform);

  // 填充vector
  result[0] = xyzrpy[0];
  result[1] = xyzrpy[1];
  result[2] = xyzrpy[2];
  result[3] = xyzrpy[3];
  result[4] = xyzrpy[4];
  result[5] = xyzrpy[5];

  message.end_pos = result;

  std::vector<double> joint_pos_vector = interfaces_ptr_->getJointPositons();
  for (int i = 0; i < 7; i++) {
    message.joint_pos[i] = joint_pos_vector[i];
  }

  std::vector<double> joint_velocities_vector = interfaces_ptr_->getJointVelocities();
  for (int i = 0; i < 7; i++) {
    message.joint_vel[i] = joint_velocities_vector[i];
  }

  std::vector<double> joint_current_vector = interfaces_ptr_->getJointCurrent();
  for (int i = 0; i < 7; i++) {
    message.joint_cur[i] = joint_current_vector[i];
  }
  // 发布消息
  joint_state_publisher_->publish(message);
}

void X5Controller::VrCmdCallback(const arm_control::msg::PosCmd::SharedPtr msg) {
  // RCLCPP_INFO(this->get_logger(), "接收到数据");
  double input[6] = {msg->x, msg->y, msg->z, msg->roll, msg->pitch, msg->yaw};
  Eigen::Isometry3d transform = solve::Xyzrpy2Isometry(input);

  interfaces_ptr_->setEndPose(transform);

  interfaces_ptr_->setArmStatus(InterfacesThread::state::END_CONTROL);

  double gripper = msg->gripper;
  if(new_version_)
    gripper *= (-3.4 / 5);
  interfaces_ptr_->setCatch(gripper);
}

void X5Controller::VrPubState() {
  // RCLCPP_INFO(this->get_logger(), "发布数据");
  auto message = arm_control::msg::PosCmd();
  // message.header.stamp = this->get_clock()->now();

  Eigen::Isometry3d transform = interfaces_ptr_->getEndPose();

  // 提取四元数和位移
  Eigen::Quaterniond quat(transform.rotation());
  Eigen::Vector3d translation = transform.translation();

  std::vector<double> xyzrpy = solve::Isometry2Xyzrpy(transform);

  // 填充vector

  message.x = xyzrpy[0];
  message.y = xyzrpy[1];
  message.z = xyzrpy[2];
  message.roll = xyzrpy[3];
  message.pitch = xyzrpy[4];
  message.yaw = xyzrpy[5];
  message.quater_x = quat.x();
  message.quater_y = quat.y();
  message.quater_z = quat.z();
  message.quater_w = quat.w();

  std::vector<double> joint_pos_vector = interfaces_ptr_->getJointPositons();
  std::vector<double> joint_velocities_vector = interfaces_ptr_->getJointVelocities();
  std::vector<double> joint_current_vector = interfaces_ptr_->getJointCurrent();

  message.gripper = joint_pos_vector[6];

  // 发布消息
  vr_joint_state_publisher_->publish(message);

  //==========================================================
  auto msg = arx5_arm_msg::msg::RobotStatus();
  msg.header.stamp = this->get_clock()->now();

  // 创建长度为6的vector
  std::array<double, 6> result;

  // 填充vector
  result[0] = xyzrpy[0];
  result[1] = xyzrpy[1];
  result[2] = xyzrpy[2];
  result[3] = xyzrpy[3];
  result[4] = xyzrpy[4];
  result[5] = xyzrpy[5];

  msg.end_pos = result;

  for (int i = 0; i < 7; i++) {
    msg.joint_pos[i] = joint_pos_vector[i];
  }

  for (int i = 0; i < 7; i++) {
    msg.joint_vel[i] = joint_velocities_vector[i];
  }

  for (int i = 0; i < 7; i++) {
    msg.joint_cur[i] = joint_current_vector[i];
  }
  // 发布消息
  joint_state_publisher_->publish(msg);
}

void X5Controller::FollowCmdCallback(const arx5_arm_msg::msg::RobotStatus::SharedPtr msg) {
  std::vector<double> target_joint_position(6, 0.0);

  for (int i = 0; i < 6; i++) {
    target_joint_position[i] = msg->joint_pos[i];
  }

  interfaces_ptr_->setJointPositions(target_joint_position);
  interfaces_ptr_->setArmStatus(InterfacesThread::state::POSITION_CONTROL);

  if (catch_control_mode_ == CatchControlMode::kPosition) {
    double pos;
    if(new_version_)
      pos = msg->joint_pos[6];
    else
      pos = msg->joint_pos[6] * 5;
    interfaces_ptr_->setCatch(pos);
  }
  else
    interfaces_ptr_->setCatchTorque(msg->joint_cur[6]);
}

void X5Controller::JointControlCallback(const arm_control::msg::JointControl::SharedPtr msg) {
  std::vector<double> target_joint_position(6,0.0);
  for (int i = 0; i < 6; i++)
    target_joint_position[i] = msg->joint_pos[i];
  interfaces_ptr_->setJointPositions(target_joint_position);
  interfaces_ptr_->setArmStatus(InterfacesThread::state::POSITION_CONTROL);

  interfaces_ptr_->setCatch(msg->joint_pos[6]);
}
}

void signalHandler(int signum) {
  rclcpp::shutdown();
}

int main(int argc, char *argv[]) {
  std::signal(SIGHUP, signalHandler);
  std::signal(SIGTERM, signalHandler);
  std::signal(SIGINT, signalHandler);

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<arx::x5::X5Controller>());
  rclcpp::shutdown();
  return 0;
}
