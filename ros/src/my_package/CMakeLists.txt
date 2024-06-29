cmake_minimum_required(VERSION 3.0.2)
project(my_package)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_generation
)

add_message_files(
  FILES
  ImagePose.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
