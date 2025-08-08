#!/usr/bin/env python3

import sys
import rospy
import math
import numpy as np
import copy
from typing import List, Dict, Set
import time

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped, PointStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import tf.transformations as tf
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_msgs.msg import ColorRGBA


class Controller:
    def __init__(self):
        # Flag to get robot positions from gazebo
        self.has_get_initial_pose = False
        self.vx = 0
        self.vy = 0
        self.stop = False  # Flag to stop robot

        # Subscribe robots' position from Gazebo
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_model_states)
        # Outside topic to stop robot's motion
        rospy.Subscriber('/stop', Bool, self.callback_stop)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

        # Set robots' positions in Gazebo
        self.pub_model_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=5)

    def set_move_step(self, step):
        self.step = step

    def set_control_rate(self, control_rate):
        self.control_rate = control_rate

    def set_robot_pose(self, pose: tuple):
        self.robot_pose = pose

    def get_robot_pose(self) -> tuple:
        return self.robot_pose

    def callback_model_states(self, data: ModelStates):
        robot_to_indices = {}
        for i, model_name in enumerate(data.name):
            if "robot_1" in model_name:  
                robot_x, robot_y = data.pose[i].position.x, data.pose[i].position.y
                orientation = data.pose[i].orientation
                quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
                euler = tf.euler_from_quaternion(quaternion)
                yaw = euler[2]
                self.set_robot_pose((robot_x, robot_y, yaw))
                if not self.has_get_initial_pose:
                    self.has_get_initial_pose = True

    def callback_stop(self, data: Bool):
        if data.data:
            self.stop = True
        else:
            self.stop = False

    def cmd_vel_callback(self, msg):
        # Extract linear and angular velocities
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        print(f"Get velocity: vx={self.vx}, vy={self.vy}")
        return

    def main_loop(self):
        self.rate = rospy.Rate(self.control_rate)  # 30 Hz
        self.dt = 1 / self.control_rate

        while not rospy.is_shutdown():
            if not self.has_get_initial_pose:
                print("Robot initial position has not been received!")
                self.rate.sleep()
                continue

            # Update robot positions
            if not self.stop:  # Whether the movement of robots is disabled
            
                curr_pose = self.get_robot_pose()

                # Calculate movement
                new_pose = (curr_pose[0] + self.vx*self.dt, curr_pose[1] + self.vy*self.dt, curr_pose[2])

                # Set new pose
                self.set_robot_pose(new_pose)

                # Publish robot pose in Gazebo
                robot_state = ModelState()
                robot_state.model_name = "robot_1"

                vehicle_roll = 0.0
                vehicle_pitch = 0.0
                vehicle_yaw = curr_pose[2]
                geo_quat = tf.quaternion_from_euler(vehicle_roll, vehicle_pitch, vehicle_yaw)
                robot_state.pose.orientation = Quaternion(*geo_quat)
                
                robot_state.pose.position.x = new_pose[0]
                robot_state.pose.position.y = new_pose[1]
                robot_state.pose.position.z = 1
                self.pub_model_state.publish(robot_state)
            
            self.rate.sleep()


def main():
    rospy.init_node('robot_controller')

    # Control rate
    control_rate = rospy.get_param('/control_rate', 30)
    # Final movement step
    step = rospy.get_param('/step', 0.2) # currently not in use

    # ModelStatesListener
    robot_controller = Controller()
    robot_controller.set_move_step(step)
    robot_controller.set_control_rate(control_rate)

    rospy.sleep(2)
    robot_controller.main_loop()
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass