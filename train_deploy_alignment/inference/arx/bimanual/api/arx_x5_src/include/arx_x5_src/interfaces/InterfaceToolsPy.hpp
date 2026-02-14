#pragma once

#include <mutex>
#include <vector>
#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Geometry> // 包含变换和位姿处理

namespace arx::x5
{
    class InterfacesToolsPy
    {
    public:
        InterfacesToolsPy(int flag);
        ~InterfacesToolsPy();
        std::vector<double> forward_kinematics_rpy(std::vector<double> joint_pos);

    private:
        class impl;
        std::unique_ptr<impl> pimpl;
    };
}