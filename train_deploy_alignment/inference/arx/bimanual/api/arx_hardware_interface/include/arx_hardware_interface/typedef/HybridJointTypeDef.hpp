#pragma once

namespace arx
{
  namespace hw_interface
  {
    typedef struct HybridJointCmd
    {
      double position;
      double velocity;
      double torque;
      double k_p;
      double k_d;
    } HybridJointCmd;

    typedef struct HybridJointStatus
    {
      double position = 0;
      double velocity = 0;
      double torque = 0;
      double current = 0;
      HybridJointStatus()
          : position(0.0), velocity(0.0), torque(0.0), current(0.0) {}
    } HybridJointStatus;
  } // namespace hw_interface
} // namespace arx