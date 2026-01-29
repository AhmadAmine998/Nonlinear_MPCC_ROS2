cmake_minimum_required(VERSION 3.5)
project(nonlinear_mpc_casadi)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

install(PROGRAMS
  scripts/Nonlinear_MPC_node.py
  scripts/Nonlinear_MPC.py
  scripts/fssim_cmd_interface.py
  scripts/fssim_track_extractor.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  params
  maps
  rviz
  scripts/fssim
  scripts/porto
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
