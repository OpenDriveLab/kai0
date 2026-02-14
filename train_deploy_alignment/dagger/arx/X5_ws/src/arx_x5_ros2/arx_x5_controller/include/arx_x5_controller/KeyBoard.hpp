#pragma once

#include "arx5_arm_msg/msg/robot_cmd.hpp"

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>

namespace arx::x5 
{
    class KeyBoardNode : public rclcpp::Node
    {
    public:
        KeyBoardNode();

        int ScanKeyBoard();
        void Update();

    private:
        rclcpp::Publisher<arx5_arm_msg::msg::RobotCmd>::SharedPtr joint_cmd_publisher_;
        rclcpp::TimerBase::SharedPtr timer_;

        arx5_arm_msg::msg::RobotCmd message_;

        int key_[3] = {0};
    };
}