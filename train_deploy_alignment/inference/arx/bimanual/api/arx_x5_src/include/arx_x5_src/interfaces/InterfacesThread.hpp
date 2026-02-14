#pragma once

#include <mutex>
#include <vector>
#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Geometry> // 包含变换和位姿处理

#include "arx_hardware_interface/typedef/HybridJointTypeDef.hpp"

namespace arx::x5
{
    class InterfacesThread
    {
    public:
        InterfacesThread(const std::string& urdf_path, const std::string &can_name, int type);
        ~InterfacesThread();

        std::vector<hw_interface::HybridJointStatus> getJointStatus();

        std::vector<double> getJointPositons();

        std::vector<double> getJointVelocities();

        std::vector<double> getJointCurrent();

        Eigen::Isometry3d getEndPose();

        void setArmStatus(int state);

        void setEndPose(Eigen::Isometry3d input);

        void setJointPositions(std::vector<double> positions);

        void setHomePositions(std::vector<double> positions);

        void setCatch(double position);

        void setCatchTorque(double torque);
        void arx_x(double arx1, double arx2, double arx3);

        std::vector<int> getErrorCode();
        /// @brief 状态枚举
        enum state
        {
            SOFT,
            GO_HOME,
            PROTECT,
            G_COMPENSATION,
            END_CONTROL,
            POSITION_CONTROL
        };

    private:
        class impl;
        std::unique_ptr<impl> pimpl;
    };
}

namespace arx::solve
{
    Eigen::Isometry3d Xyzrpy2Isometry(double input[6]);
    std::vector<double> Isometry2Xyzrpy(Eigen::Isometry3d pose);
}
