cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 17)

project(data_play)

## Find Gazebo and ROS dependencies
find_package(gazebo REQUIRED)
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
)

add_message_files(
  FILES
  ModelInfo.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## Include directories for Gazebo and ROS
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${roscpp_INCLUDE_DIRS}
)

## Link directories for Gazebo
link_directories(${GAZEBO_LIBRARY_DIRS})

## Add the Gazebo plugin source file
add_library(ActorTrajectoryROSPlugin SHARED src/ActorTrajectoryROSPlugin.cpp)
add_library(TrajectoryFollowerPlugin SHARED src/TrajectoryFollowerPlugin.cpp)
add_library(ActorTriggerPlugin SHARED src/actor_trigger.cpp)
add_library(ModelFollowerPlugin SHARED src/ModelFollowerPlugin.cpp)
add_library(ActorCollisionsPlugin SHARED src/ActorCollisionsPlugin.cc)
## Link the Gazebo plugin against necessary libraries
target_link_libraries(ActorTrajectoryROSPlugin
  TrajectoryFollowerPlugin
  ActorTriggerPlugin
  ModelFollowerPlugin
  ActorCollisionsPlugin
  ${GAZEBO_LIBRARIES}
  ${roscpp_LIBRARIES}
)

