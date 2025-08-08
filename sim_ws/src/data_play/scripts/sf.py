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

from check_visibility import human_scoring,get_visible_region

# Social False
# activate base planner: default=True

class social_force:
    def __init__(self,obs_list,radius):
        self.obs_list=obs_list
        self.A=1.5
        self.B=1.0
        self.radius=radius
        self.v_pref=1.0
        self.KI = 1.0
    
    def point_to_line_distance_with_point(self, x0, y0, x1, y1, x2, y2):
        # Line coefficients: Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # Perpendicular distance
        distance = abs(A * x0 + B * y0 + C) / math.sqrt(A**2 + B**2)

        # Finding the closest point on the line
        # Parametric equation of the line
        if A**2 + B**2 == 0:  # Avoid division by zero for degenerate line
            x_closest = x0
            y_closest = y0
            return distance, x_closest, y_closest

        # Perpendicular projection formula
        x_closest = (B * (B * x0 - A * y0) - A * C) / (A**2 + B**2)
        y_closest = (A * (-B * x0 + A * y0) - B * C) / (A**2 + B**2)

        dist1=(x_closest-x1)**2+(y_closest-y1)**2
        dist2=(x_closest-x2)**2+(y_closest-y2)**2

        dist_1_2=(x1-x2)**2+(y1-y2)**2

        if dist1>dist_1_2 or dist2>dist_1_2:
            x_closest=100000
            y_closest=100000
            distance=100000

        return distance, x_closest, y_closest

    def predict(self,goal,state,humans,mapping,cv_pref): #TODO humans are what?
        self.v_pref=cv_pref
        delta_position=goal-state[0:2]
        delta_x=delta_position[0]
        delta_y=delta_position[1]
        dist_to_goal = np.linalg.norm(delta_position)
        desired_vx = (delta_x / dist_to_goal) * self.v_pref
        desired_vy = (delta_y / dist_to_goal) * self.v_pref
        curr_delta_vx = self.KI * (desired_vx - state[2])
        curr_delta_vy = self.KI * (desired_vy -state[3])
        A=self.A
        B=self.B
        interaction_vx = 0
        interaction_vy = 0
        min_dist_to_human=10000
        min_dist_to_obs=10000
        vx_human=0
        vy_human=0
        vx_obs=0
        vy_obs=0

        my_position=state[0:2]
        min_dist_to_human = float('inf')

        for human in humans:
            human_id = int(human[0])  # Extract the human's ID
            other_human_pos = human[1:3]  # Extract (pos_x, pos_y)
            
            human_radius=1

            if human_id not in mapping:
                continue  # Skip if ID is not in mapping
            if mapping[human_id]==0:
                human_radius=0.5    
            elif mapping[human_id]==1:
                human_radius=0.5
            elif mapping[human_id]==2:
                human_radius=0.5

            delta_x = my_position[0] - other_human_pos[0]
            delta_y = my_position[1] - other_human_pos[1]
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)

            if dist_to_human == 0:  # Avoid division by zero
                continue

            if min_dist_to_human > dist_to_human:
                min_dist_to_human = dist_to_human
                vx_human = A * np.exp((self.radius + human_radius - dist_to_human) / B) * (delta_x / dist_to_human)
                vy_human = A * np.exp((self.radius + human_radius - dist_to_human) / B) * (delta_y / dist_to_human)

        interaction_vx += vx_human
        interaction_vy += vy_human

        for obstacles in self.obs_list:
            if obstacles.shape[0]==1:
                delta_x = my_position[0] - obstacles[0,0]
                delta_y = my_position[1] - obstacles[0,1]
                dist_to_obs=np.sqrt(delta_x**2 + delta_y**2)
                if min_dist_to_obs> dist_to_obs:
                    min_dist_to_obs=dist_to_obs
                    vx_obs=3*A * np.exp((self.radius - dist_to_obs) / B) * (delta_x / dist_to_obs)
                    vy_obs=3*A * np.exp((self.radius - dist_to_obs) / B) * (delta_y / dist_to_obs)
            else:
                for row_index in range(0,obstacles.shape[0]-1):
                    x0=my_position[0]
                    y0=my_position[1]
                    x1=obstacles[row_index,0]
                    y1=obstacles[row_index,1]
                    x2=obstacles[row_index+1,0]
                    y2=obstacles[row_index+1,1]
                    dist_to_obs,x_closest,y_closest=self.point_to_line_distance_with_point(x0,y0,x1,y1,x2,y2)
                    delta_x = my_position[0] - x_closest
                    delta_y = my_position[1] - y_closest
                    dist_to_obs=np.sqrt(delta_x**2 + delta_y**2)
                    if min_dist_to_obs> dist_to_obs:
                        min_dist_to_obs=dist_to_obs
                        vx_obs=3*A * np.exp((self.radius - dist_to_obs) / B) * (delta_x / dist_to_obs)
                        vy_obs=3*A * np.exp((self.radius - dist_to_obs) / B) * (delta_y / dist_to_obs)
        interaction_vx+=vx_obs*0.1
        interaction_vy+=vy_obs*0.1
        total_delta_vx = (curr_delta_vx + interaction_vx) 
        total_delta_vy = (curr_delta_vy + interaction_vy) 
        new_vx = state[2] + total_delta_vx
        new_vy = state[3] + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])
        if act_norm > self.v_pref:
            return np.array([new_vx / act_norm * self.v_pref, new_vy / act_norm * self.v_pref])
        else:
            return np.array([new_vx, new_vy])
    

class OurPlanner:
    def __init__(self,goal,obs_list):

        self.default=True
        self.robot_range = 10
        self.robot_radius=1.0        
        self.previous_leader = None

        self.robot_speed_max = 1.4 

        self.goal=goal
        self.obs_list=obs_list
        self.history_list=[]
        self.base_controller=social_force(self.obs_list,self.robot_radius)
        self.list_length=25
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end
        self.time_step=0.11

    def predict(self,state,human_step,mask,laser_scan):
        
        global_goal_x=self.goal[0]
        global_goal_y=self.goal[1]

        if self.default:
            print("##### Using default SF planner #####")
            new_gx = global_goal_x
            new_gy = global_goal_y
            command_v_pref = self.robot_speed_max

        comand_goal=np.array([new_gx,new_gy])
        # print("Command Goal:", comand_goal)
        # print("Robot state:",state)
        action=self.base_controller.predict(comand_goal,state,human_step,mask,command_v_pref)

        self.following_id_vis = 0
        invisible_list = []
        visible_region_edges = np.array([])

        return self.following_id_vis, action, comand_goal, invisible_list, visible_region_edges
    
    def clear_buffer(self):
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end


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

        self.planner=OurPlanner(self.goal_pos,self.obs_list)

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
            track_id,action,subgoal,invis_index,edges_visible_region=self.planner.predict(state_now,model_temp,self.id_mask,self.laser_scan)
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
            invis_id_msg.data=invis_index
            # print(type(invis_index))
            # print(invis_index)
            self.invisable_pub.publish(invis_id_msg)

            msg=Float32MultiArray()
            msg.data = [track_id,subgoal[0],subgoal[1],self.robot_state[0],self.robot_state[1]]

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
