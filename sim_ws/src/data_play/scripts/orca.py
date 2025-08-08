#!/usr/bin/env python3

import rospy
import time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import json
import threading
import numpy as np
from sensor_msgs.msg import LaserScan  # Laser scan message
from data_play.msg import ModelInfo
import math
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

# ORCA
import rvo2


class orca:
    def __init__(self,goal,obs_list):
        self.goal=goal
        self.obs_list=obs_list
        self.neighbor_dist=10
        self.max_neighbors=20
        self.time_horizon=0.6
        self.time_horizon_obs=2
        self.max_speed=1.4
        self.sim=None
        
    def predict(self,state,human_step,mask,laser_scan):
        self_state = state
        params=self.neighbor_dist, self.max_neighbors, self.time_horizon,self.time_horizon_obs
        if self.sim is not None and  self.sim.getNumAgents() != len(human_step)+1:
            del self.sim
            self.sim=None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(1.0/9,*params,1.0,2.0)
            for obs in self.obs_list:
                # Convert NumPy array to list of tuples
                obstacle_points = [tuple(point) for point in obs[:-1,:]]
                # Add the obstacle to the simulation
                o1=self.sim.addObstacle(obstacle_points)
            self.sim.processObstacles()
            self.sim.addAgent((self_state[0], self_state[1]), *params, 1,self.max_speed, (self_state[2], self_state[3]))
            for human_state in human_step:
                self.sim.addAgent((human_state[1], human_state[2]), *params, 0.5, 1.4, (human_state[3], human_state[4]))
        else:
            self.sim.setAgentMaxSpeed(0,1.4)
            self.sim.setAgentPosition(0, (self_state[0],self_state[1]))
            self.sim.setAgentVelocity(0, (self_state[2],self_state[3]))
            self.sim.setAgentRadius(0,1)
            for i, human_state in enumerate(human_step):
                self.sim.setAgentPosition(i + 1, (human_state[1], human_state[2]))
                self.sim.setAgentVelocity(i + 1, (human_state[3], human_state[4]))

        velocity = np.array((self.goal[0] - self_state[0], self.goal[1] - self_state[1]))
        speed = np.linalg.norm(velocity)
        if speed:
            pref_vel = velocity / speed * self.max_speed
        else:
            pref_vel = velocity / speed if speed > self.max_speed else velocity 
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(human_step):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))
        self.sim.doStep()
        action=np.array(self.sim.getAgentVelocity(0))
        print(action)
        return action
    def clear_buffer(self):
        return
        


class Robot:

    def __init__(self):
        self.lock=threading.Lock()
        rospy.init_node('robot_listener', anonymous=True)
        scene=rospy.get_param("scene","nexus_2_0")
        self.robot_horizon=15.0
        self.scene_path = '/root/sim_ws/src/data_play/dataset/scene_config_30/' + scene + '.json'

        with open(self.scene_path, 'r') as f:
            self.scene_config = json.load(f)  # Correct way to load JSON

        robot_pos_goal = np.array(self.scene_config["robot_start_end"])

        self.start_pos=robot_pos_goal[0:2]

        self.goal_pos=robot_pos_goal[2:4]

        self.gaol_range=0.3

        self.obs_list=[]

        for obs in self.scene_config["obstacles"]:
            self.obs_list.append(np.array(obs))

        self.planner=orca(self.goal_pos,self.obs_list)

        self.start_mission=0

        rospy.wait_for_service('/gazebo/set_model_state')

        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.robot_state=None
        self.id_mask=None
        self.visable_humans=None

        self.laser_point=None
        
        self.laser_scan=[]

        # Subscribers
        self.model_info_sub = rospy.Subscriber('/gazebo/model_info', ModelInfo, self.model_info_callback)
        self.odom_sub = rospy.Subscriber('/robot_1/odom', Odometry, self.odom_callback)
        self.mission_sub = rospy.Subscriber('/env_control', Int32, self.mission_callback)  # New subscriber for env_control
        self.laser_sub = rospy.Subscriber('/robot_1/laser_scan', LaserScan, self.laser_callback)
        self.invisable_pub=rospy.Publisher('/invisable_id',Int32MultiArray,queue_size=1)
        self.vis_pub = rospy.Publisher("/vis_array_topic", Float32MultiArray, queue_size=10)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        rospy.loginfo("Robot Node Initialized: Subscribed to /gazebo/model_info and /robot_1/odom, Publishing to /cmd_vel")
        
        self.edges_visible_region = np.array([])  # used to publish visible region to rviz
        self.pubVisibleEdges = rospy.Publisher('/visibleEdges', Marker, queue_size=2)

    def pubVisibleRegion(self):
        marker = Marker()
        marker.header.frame_id = "robot_1/base_link"  # adjust the frame as needed (e.g., "base_link")
        marker.header.stamp = rospy.Time.now()
        marker.ns = "edges"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Set marker properties
        marker.scale.x = 0.05  # Line width
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Fully opaque

        # Populate marker.points by adding each edge's start and end points.
        marker.points = []
        for i in range(len(self.edges_visible_region)):
            # Get start and end coordinates from the i-th column.
            point1, point2 = self.edges_visible_region[i]
            start_point = Point(point1[0], point1[1], 0)  # z set to 0
            end_point   = Point(point2[0], point2[1], 0)

            marker.points.append(start_point)
            marker.points.append(end_point)
        self.pubVisibleEdges.publish(marker)
    
    def laser_callback(self, msg):
        self.laser_scan = msg.ranges
        # self.laser_scan = [] # do not use laser scan
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.range_max = msg.range_max

    def mission_callback(self,msg):
        if msg.data==0:                
            model_state = ModelState()
            model_state.model_name = 'robot_1'
            model_state.pose.position.x = self.start_pos[0]
            model_state.pose.position.y = self.start_pos[1]
            model_state.pose.position.z = 0.0
            response = self.set_state(model_state)
            self.planner.clear_buffer()
        with self.lock:
            self.start_mission=msg.data

    def model_info_callback(self, msg):
        if self.id_mask is None:
            self.id_mask={}
            for i in range(len(msg.ids)):  # Loop through all models
                model_id = msg.ids[i]  # Use index as ID (replace if needed)
                model_tag = msg.tags[i]  # Example: Using model name as a "tag"
                self.id_mask[model_id] = model_tag  # Store in dict
        # rospy.loginfo(f"id_mask: {self.id_mask}")
        if self.robot_state is None:
            return
        with self.lock:
            state_now=self.robot_state[0:4]
        model_temp=[]
        for i in range(len(msg.ids)):  # Loop through all models
            model_id = msg.ids[i]  # Assuming id is based on index (replace if needed)
            px, py = msg.position[i].x, msg.position[i].y
            vx, vy = msg.velocity[i].x, msg.velocity[i].y
            dx=px-state_now[0]
            dy=py-state_now[1]
            if np.sqrt(dx**2+dy**2)<self.robot_horizon:
                model_temp.append([model_id, px, py, vx, vy])
        self.visable_humans=model_temp
        
        if self.start_mission==1:
            action=self.planner.predict(state_now,model_temp,self.id_mask,self.laser_scan)
            if np.linalg.norm(self.goal_pos - state_now[0:2])< self.gaol_range:
                action=np.array([0,0])
                self.start_mission=0
                self.planner.clear_buffer()
            # rospy.loginfo(f"track_id: {track_id},action: {action}")

            # Create Twist message
            cmd_msg = Twist()
            cmd_msg.linear.x = action[0]
            cmd_msg.linear.y = action[1]
            self.cmd_vel_pub.publish(cmd_msg)

            invis_id_msg=Int32MultiArray()
            invis_id_msg.data=[]
            # print(type(invis_index))
            # print(invis_index)
            self.invisable_pub.publish(invis_id_msg)

            msg=Float32MultiArray()
            msg.data = [2000,self.goal_pos[0],self.goal_pos[1],self.robot_state[0],self.robot_state[1]]

            self.vis_pub.publish(msg)

        else:
            invis_id_msg=Int32MultiArray()
            invis_id_msg.data=[]
            self.invisable_pub.publish(invis_id_msg)
            msg=Float32MultiArray()
            msg.data = [-1,self.goal_pos[0],self.goal_pos[1],self.robot_state[0],self.robot_state[1]]
            self.vis_pub.publish(msg)
            cmd_msg = Twist()
            cmd_msg.linear.x = 0
            cmd_msg.linear.y = 0
            self.cmd_vel_pub.publish(cmd_msg)
            model_state = ModelState()
            model_state.model_name = 'robot_1'
            model_state.pose.position.x = self.start_pos[0]
            model_state.pose.position.y = self.start_pos[1]
            model_state.pose.position.z = 0.0
            response = self.set_state(model_state)
            self.planner.clear_buffer()

    def odom_callback(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        with self.lock:
            self.robot_state = [px, py, vx, vy]
        # print(f"robot state {px},{py}")
        

if __name__ == '__main__':
    try:
        robot = Robot()
        rospy.spin()  # Keep the node running, waiting for callbacks
    except rospy.ROSInterruptException:
        pass
