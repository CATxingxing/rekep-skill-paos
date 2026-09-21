# Robotiq 2F-85 URDF

这是 ROS-Industrial `robotiq_2f_85_gripper_visualization` 的 2F-85 leaf 描述，
含 Xacro、展开的 `robotiq_arg2f_85_model.urdf` 和 DAE/STL 网格。来源为：

- <https://github.com/ros-industrial/robotiq/tree/kinetic-devel/robotiq_2f_85_gripper_visualization>
- revision `45196f6558fe8ba9d89bc8a105396c68c3e7e892`
- BSD-3-Clause，许可证见 `UPSTREAM_LICENSE`

它描述机械结构和 mimic joints；真实电气传输依现场的 URCap、Modbus TCP/RTU 或
ROS 驱动决定，不由该 URDF 文件推断。
