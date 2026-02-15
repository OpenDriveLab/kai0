# AgileX Robotic Arm

[CN](README.MD)

![ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange.svg)

Test:

|PYTHON |STATE|
|---|---|
|![noeti](https://img.shields.io/badge/ros-noetic-blue.svg)|![Pass](https://img.shields.io/badge/Pass-blue.svg)|

## Installation

### Install dependencies

```shell
pip3 install python-can
```

```shell
pip3 install piper_sdk
```

## Quick Start

### Enable CAN module

First, you need to set up the shell script parameters.

#### Single robotic arm

##### PC with Only One USB-to-CAN Module Inserted

- ##### Use the `can_activate.sh`  

Directly run:

```bash
bash can_activate.sh can0 1000000
```

##### PC with Multiple USB-to-CAN Modules Inserted

- ##### Use the `can_activate.sh`  script

Disconnect all CAN modules.

Only connect the CAN module linked to the robotic arm to the PC, and then run the script.

```shell
sudo ethtool -i can0 | grep bus
```

and record the `bus-info` value, for example, `1-2:1.0`.

**Note: Generally, the first inserted CAN module defaults to `can0`. If the CAN interface is not found, use `bash find_all_can_port.sh` to check the CAN names corresponding to the USB addresses.**

Assuming the recorded `bus-info` value from the above operation is `1-2:1.0`.

Then execute the command to check if the CAN device has been successfully activated.

```bash
bash can_activate.sh can_piper 1000000 "1-2:1.0"
```

**Note: This means that the CAN device connected to the USB port with hardware encoding `1-2:1.0` is renamed to `can_piper`, set to a baud rate of 1,000,000, and activated.**

Then run`ifconfig` to check if `can_piper` appears. If it does, the CAN module has been successfully configured.

#### Two Pairs of Robotic Arms (Four Arms)

For four robotic arms, which means two pairs of master-slave robotic arms:

- ##### Use the `can_config.sh` script here

In the `can_config.sh` , set the `EXPECTED_CAN_COUNT` parameter to `2`, as four robotic arms use two CAN modules.

Then, insert one of the two CAN modules (usually the one connected to the left arm) into the PC alone and execute the script.

```shell
sudo ethtool -i can0 | grep bus
```

and record the `bus-info` value, for example, `1-2:1.0`.

Next, insert the second CAN module, ensuring it is connected to a different USB port than the one used previously, and then execute the script.

```shell
sudo ethtool -i can1 | grep bus
```

**Note: Generally, the first inserted CAN module defaults to `can0`, and the second one to `can1`. If the CAN interfaces are not found, use `bash find_all_can_port.sh` to check the CAN names corresponding to the USB addresses.**

Assuming the recorded `bus-info` values are `1-2:1.0` and `1-4:1.0`, replace the parameters inside the double quotes in `USB_PORTS["1-9:1.0"]="can_left:1000000"` with `1-2:1.0`.

Similarly, update the other one

`USB_PORTS["1-5:1.0"]="can_right:1000000"` -> `USB_PORTS["1-4:1.0"]="can_right:1000000"`

**Note: This means that the CAN device connected to the USB port with hardware encoding `1-2:1.0` is renamed to `can_left`, set to a baud rate of 1,000,000, and activated.**

Then execute `bash can_config.sh` and check the terminal output to see if the activation was successful.

Afterward, run `ifconfig` to verify if `can_left` and `can_right` appear. If they do, the CAN modules have been successfully configured.

### Running the Node

#### Single Robotic Arm

Node name: `piper_ctrl_single_node.py`

param

```shell
can_port:he name of the CAN route to open.
auto_enable: Whether to automatically enable the system. If True, the system will automatically enable upon starting the program.
#  Set this to False if you want to manually control the enable state. If the program is interrupted and then restarted, the robotic arm will maintain the state it had during the last run.
# If the arm was enabled, it will remain enabled after restarting.
# If the arm was disabled, it will remain disabled after restarting.
girpper_exist:Indicates if there is an end-effector gripper. If True, the gripper control will be enabled.
rviz_ctrl_flag: Whether to use RViz to send joint angle messages. If True, the system will receive joint angle messages sent by rViz.
# Since the joint 7 range in RViz is [0,0.04], but the actual gripper travel is 0.08m, joint 7 values sent by RViz will be multiplied by 2 when controlling the gripper.
```

`start_single_piper_rviz.launch`:

```xml
<launch>
  <arg name="can_port" default="can0" />
  <arg name="auto_enable" default="true" />
  <include file="$(find piper_description)/launch/display_xacro.launch"/>
  <!-- Start robotic arm node-->
  <node name="piper_ctrl_single_node" pkg="piper" type="piper_ctrl_single_node.py" output="screen">
    <param name="can_port" value="$(arg can_port)" />
    <param name="auto_enable" value="$(arg auto_enable)" />
    <param name="rviz_ctrl_flag" value="true" />
    <param name="girpper_exist" value="true" />
    <remap from="joint_ctrl_single" to="/joint_states" />
  </node>
</launch>
```

`start_single_piper.launch`:

```xml
<launch>
  <arg name="can_port" default="can0" />
  <arg name="auto_enable" default="true" />
  <!-- <include file="$(find piper_description)/launch/display_xacro.launch"/> -->
  <!-- Start robotic arm node -->
  <node name="piper_ctrl_single_node" pkg="piper" type="piper_ctrl_single_node.py" output="screen">
    <param name="can_port" value="$(arg can_port)" />
    <param name="auto_enable" value="$(arg auto_enable)" />
    <param name="rviz_ctrl_flag" value="true" />
    <param name="girpper_exist" value="true" />
    <remap from="joint_ctrl_single" to="/joint_states" />
  </node>
</launch>
```

Start control node

```shell
# Start node
roscore
rosrun piper piper_ctrl_single_node.py _can_port:=can0 _mode:=0
# Start launch
roslaunch piper start_single_piper.launch can_port:=can0 auto_enable:=true
# Or,the node can be run with default parameters
roslaunch piper start_single_piper.launch
# You can also use RViz to enable control by adjusting the parameters as described above.
roslaunch piper start_single_piper_rviz.launch
```

#### Multiple Robotic Arms

##### Read master-slave arm messages without controlling the slave arm

`start_ms_piper.launch` 

```xml
<launch>
  <!-- Define the `mode` parameter, with a default value of `0`. This mode will read the master-slave arm messages and forward them to ROS. -->
  <arg name="mode" default="0" />
  <arg name="auto_enable" default="true" />
  <!-- Start the left-side robotic arm node -->
    <node name="$(anon piper_left)" pkg="piper" type="piper_start_ms_node.py" output="screen">
      <param name="can_port" value="can_left" />
      <param name="mode" value="$(arg mode)" />
      <param name="auto_enable" value="$(arg auto_enable)" />
      <remap from="/puppet/joint_states" to="/puppet/joint_left" />
      <remap from="/master/joint_states" to="/master/joint_left" />
    </node>

  <!-- Start the right-side robotic arm node -->
    <node name="$(anon piper_right)" pkg="piper" type="piper_start_ms_node.py" output="screen">
      <param name="can_port" value="can_right" />
      <param name="mode" value="$(arg mode)" />
      <param name="auto_enable" value="$(arg auto_enable)" />
      <remap from="/puppet/joint_states" to="/puppet/joint_right" />
      <remap from="/master/joint_states" to="/master/joint_right" />
    </node>
</launch>
```

Start the node, mainly using `launch` to start both the left and right side arms.

```shell
# When collecting data, read the master-slave arm messages. In this case, the `auto_enable` parameter setting can be omitted.
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=true
```

and run

```shell
rostopic list
```

There are several topics available, such as：

```shell
/arm_status
/master/joint_left
/master/joint_right
/puppet/joint_left
/puppet/joint_right
```

Among these topics, `/master/joint_left`, `/master/joint_right`, `/puppet/joint_left`, and `/puppet/joint_right` can be used to read messages from the master and slave arms.

##### Control the Slave Arm via the Node

**First, disconnect the main arm's connector!!!**

```shell
# To control both slave arms, you can remove the `auto_enable` parameter setting.
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```

and run

```shell
rostopic list
```

There are several topics:

```shell
/arm_status
/enable_flag
/master/joint_left
/master/joint_right
/puppet/joint_left
/puppet/joint_right
```

`/enable_flag` is used to enable the robotic arm. By default, automatic enabling is turned on in the `launch` file. If you want to disable automatic enabling, send `true` to this topic to enable the arm.

The topics `/master/joint_left` and `/master/joint_right` are used for external control, which in turn controls the slave arms.

**How to verify enabling without terminal output: After sending the enable message, manually move the robotic arm. If the joints cannot be moved, it indicates that the joints are powered and enabling was successful.**

## Note

- You need to activate the CAN device and set the correct baud rate before you can read messages from or control the robotic arm.
- **To control two slave robotic arms using ROS, you must first disconnect the main arm's connector!!!**
- When the node parameter `mode` is set to `1`, the topics `/master/joint_left`, `/master/joint_right`, `/puppet/joint_left`, and `/puppet/joint_right` are data topics only. You only need to read from them. Typically, in this mode, the CAN communication between the master and slave arms is established.
- When the node parameter `mode` is set to `0`, `/master/joint_left` and `/master/joint_right` are topics for controlling the slave arms, while `/puppet/joint_left` and `/puppet/joint_right` are for feedback from the slave arms. To send movement commands to the slave arms via `/master/joint_left` and `/master/joint_right`, you must first send `true` to the `/enable_flag` topic.

### piper Custom Messages

ros package `piper_msgs`

 **Robotic Arm Status Feedback Message**: Corresponds to the feedback message with `id=0x2A1` in the CAN protocol.

`PiperStatusMsg.msg`

```c
uint8 ctrl_mode
uint8 arm_status
uint8 mode_feedback
uint8 teach_status
uint8 motion_status
uint8 trajectory_num
int64 err_code
bool joint_1_angle_limit
bool joint_2_angle_limit
bool joint_3_angle_limit
bool joint_4_angle_limit
bool joint_5_angle_limit
bool joint_6_angle_limit
bool communication_status_joint_1
bool communication_status_joint_2
bool communication_status_joint_3
bool communication_status_joint_4
bool communication_status_joint_5
bool communication_status_joint_6
```

End-effector pose control: Note that some singularities may be unreachable.

`PosCmd.msg`

```c
float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
float64 gripper
int32 mode1
int32 mode2
```

## Simulation

`display_xacro.launch` open rviz

After running, the `/joint_states` topic will be published. You can view it by /joint_states:

![ ](./asserts/pictures/tostopic_list.jpg)

Two windows will appear simultaneously as follows. The slider values correspond to the `/joint_states` values. Dragging the sliders will change these values, and the model in rviz will update accordingly.

![ ](./asserts/pictures/piper_rviz.jpg)
