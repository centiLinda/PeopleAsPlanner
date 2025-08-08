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

    def predict(self,goal,state,humans,mapping,cv_pref):
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

        self.default=False
        self.robot_range = 10
        self.robot_radius=1.0        
        self.previous_leader = None

        #################
        # Tunable Param #-----------------------------------
        #################
        # agent
        self.human_radius = 0.8  # distance to keep when following, not the radius in planner
        self.human_speed_ideal = 1.4
        self.human_speed_min = 0.6
        self.robot_speed_max = 1.4  # avg human walking speed

        # score function
        self.goal_score_steps = 10
        self.catchup_threshold = 1.5
        self.catchup_speed = 2.0
        self.position_penalty = -1.75

        # weight
        self.weight_goal = 1.0
        self.weight_velocity = 0.5
        self.weight_position = 1.0
        self.current_leader_bias = 0.05

        # group identification
        self.min_distance_threshold = 1.5
        self.distance_threshold = 2.0
        self.velocity_threshold = 0.5
        
        # visibility
        self.inflate_radius = 0.5
        #-----------------------------------------------------

        self.goal=goal
        self.obs_list=obs_list
        self.history_list=[]
        self.base_controller=social_force(self.obs_list,self.robot_radius)
        self.list_length=25
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end
        self.time_step=0.11

    def predict(self,state,human_step,mask,laser_scan):
        
        # state [px py vx vy]
        while(len(self.state_buffer)<self.list_length):
            self.state_buffer.append(state)
        while(len(self.human_buffer)<self.list_length):
            self.human_buffer.append(human_step)
        global_goal_x=self.goal[0]
        global_goal_y=self.goal[1]
        robot_goal_vector = (global_goal_x - state[0], global_goal_y - state[1])
        goal_distance = math.sqrt(robot_goal_vector[0]**2 + robot_goal_vector[1]**2)

        #########################
        # Leader Identification #-----------------------------------------------------------------
        #########################
        # ID is not in same order as human_step
        neighbor_traj = {}
        for timestep in self.human_buffer:
            for human in timestep:
                id, px, py, vx, vy = human
                if any(agent[0] == id for agent in human_step): # only add visible human
                    if id not in neighbor_traj:
                        neighbor_traj[id] = []
                    neighbor_traj[id].append([px, py, vx, vy])
                    
        ########################
        # Group Identification #
        ########################
        groups = []
        visited = set()

        for i, human_a in enumerate(human_step):
            if human_a[0] in visited:
                continue
            group = [human_a]
            visited.add(human_a[0])

            for j, human_b in enumerate(human_step):
                if human_b[0] in visited or human_a[0] == human_b[0]:
                    continue
                distance = math.sqrt((human_a[1] - human_b[1])**2 + (human_a[2] - human_b[2])**2)
                velocity_similarity = math.sqrt((human_a[3] - human_b[3])**2 + (human_a[4] - human_b[4])**2)
                if (
                    (distance <= self.distance_threshold and velocity_similarity <= self.velocity_threshold)
                ):
                    group.append(human_b)
                    visited.add(human_b[0])

            if len(group) > 1:  # group has 2 or more humans
                groups.append(group)
        # print(f"Identified groups: {[[h[0] for h in group] for group in groups]}")

        ####################
        # Visibility Check #
        ####################
        group_list = [[agent[0] for agent in group] for group in groups]  
        
        robot_pose = [state[0], state[1], 0.0] # dummy yaw
        human_radius = [self.inflate_radius for i in range(len(human_step))] # inflate by robot passing radius
        human_scores_list = human_scoring(laser_scan, human_step, robot_pose, human_radius)
        visible_region_edges = np.array([])

        invisible_list = []
        for k, score in enumerate(human_scores_list):
            if score < 0:
                invisible_list.append(human_step[k][0])
        # print(f"Invisible: {invisible_list}")

        ###########
        # 1. Goal # heading cosine similarity -1 ~ 1
        ###########
        scores_goal = {}

        for ped_id, trajectory in neighbor_traj.items():
            num_steps = min(self.goal_score_steps, len(trajectory))
            avg_heading_vector = [0, 0]
            goal_vector = (global_goal_x - trajectory[-1][0], global_goal_y - trajectory[-1][1])
            for i in range(-1, -1 - num_steps, -1):  # Iterate backward through the trajectory
                avg_heading_vector[0] += trajectory[i][2]  # vx
                avg_heading_vector[1] += trajectory[i][3]  # vy
            avg_heading_magnitude = math.sqrt(avg_heading_vector[0]**2 + avg_heading_vector[1]**2)
            goal_magnitude = math.sqrt(goal_vector[0]**2 + goal_vector[1]**2)
            
            if avg_heading_magnitude > 0 and goal_magnitude > 0:
                avg_heading_vector = (avg_heading_vector[0] / avg_heading_magnitude, avg_heading_vector[1] / avg_heading_magnitude)
                goal_vector = (goal_vector[0] / goal_magnitude, goal_vector[1] / goal_magnitude)
            
                dot_product = avg_heading_vector[0] * goal_vector[0] + avg_heading_vector[1] * goal_vector[1]

                # Only consider scores within the +/- 45-degree range
                if dot_product >= 0.5:  # cos(45 degrees) ≈ 0.707
                    scores_goal[ped_id] = dot_product
                else:
                    scores_goal[ped_id] = -10
            else:
                scores_goal[ped_id] = -10

        ###############
        # 2. Velocity # 0 ~ 1, 0 if very high speed, negative if too slow, -10 if stationary
        ###############
        scores_velocity = {}
        human_speeds = {}
        ideal_speed = self.human_speed_ideal
        stationary_human = []
        for ped_id, trajectory in neighbor_traj.items():
            total_distance = 0
            steps = min(10, len(trajectory)) # avg over the last 10 steps
            for i in range(-1, -1 - steps, -1):
                total_distance += math.sqrt(trajectory[i][2]**2 + trajectory[i][3]**2)
            avg_speed = total_distance / steps
            human_speeds[ped_id] = math.sqrt(trajectory[-1][2]**2 + trajectory[-1][3]**2) # current speed

            if avg_speed < self.human_speed_min:
                scores_velocity[ped_id] = (avg_speed - ideal_speed) / ideal_speed
                if avg_speed < 0.1: # do not follow stationary human
                    scores_velocity[ped_id] = -10
                    stationary_human.append(ped_id)
            else:
                scores_velocity[ped_id] = max(0, 1 - (abs(avg_speed - ideal_speed) / ideal_speed))

        ###############
        # 3. Position # 0 ~ 1, -1.75 if behind robot
        ###############
        scores_position = {}
        hr_distance = {}
        hr_vector = {}
        for ped_id, trajectory in neighbor_traj.items():
            human_pos = trajectory[-1][:2]  # (px, py)
            human_vector = (human_pos[0] - state[0], human_pos[1] - state[1])
            distance = math.sqrt(human_vector[0]**2 + human_vector[1]**2)
            hr_distance[ped_id] = distance
            hr_vector[ped_id] = human_vector
            human_vector = (human_vector[0] / distance, human_vector[1] / distance)
            robot_heading = robot_goal_vector
            
            r_heading_mag = math.sqrt(robot_heading[0]**2 + robot_heading[1]**2)
            robot_heading = (robot_heading[0] / r_heading_mag, robot_heading[1] / r_heading_mag)
            
            dot_product = human_vector[0] * robot_heading[0] + human_vector[1] * robot_heading[1]
            if dot_product >= 0.5:  # within +/-60 degree
                scores_position[ped_id] = (dot_product + max(0, 1 - (distance / self.robot_range))) / 2
            else:  # behind
                scores_position[ped_id] = self.position_penalty

        #########
        # Total #
        #########
        total_scores = {}
        for ped_id in neighbor_traj.keys():
            score_goal = scores_goal.get(ped_id, -1)
            score_velocity = scores_velocity.get(ped_id, -1)
            score_position = scores_position.get(ped_id, -1)
            # filter out invisible humans
            if ped_id not in invisible_list:
                total_scores[ped_id] = (self.weight_goal * score_goal + 
                                        self.weight_velocity * score_velocity + 
                                        self.weight_position * score_position)
            
            print(f"ID: {ped_id:>1} | goal: {score_goal:>5.1f} | vel: {score_velocity:>5.1f} | pos: {score_position:>5.1f}")

        # Favour current leader to avoid fluctuation
        for human in total_scores:
            if human == self.previous_leader:
                total_scores[human] += self.current_leader_bias
                continue
                
        # check if we should follow leaders
        if (
            total_scores # exist neighbors
            and any(score > 0 for score in total_scores.values()) # exist good leader
            and goal_distance > 1.0 # still far from goal
        ):
            leader_ID = max(total_scores, key=total_scores.get)
            print(f"Leader ID: {leader_ID}     distance: {goal_distance:.2f}")

            #----If leader belongs to a group-----------------------------------------
            def get_closest_human_in_group(group, robot_x, robot_y):
                closest_human = None
                min_distance = float('inf')
                for human in group:
                    human_px, human_py = human[1], human[2]
                    distance = math.sqrt((robot_x - human_px)**2 + (robot_y - human_py)**2)
                    if distance < min_distance:
                        closest_human = human
                        min_distance = distance
                return closest_human

            leader_group = None
            for group in groups:
                if any(human[0] == leader_ID for human in group):
                    leader_group = group
                    break
            if leader_group:
                closest_human = get_closest_human_in_group(leader_group, state[0], state[1])
                if (
                    closest_human[0] != leader_ID
                    and closest_human[0] not in invisible_list # is visible
                    and total_scores[closest_human[0]] > 0 # is ok leader
                ):
                    leader_ID = closest_human[0]
                    print(f"Switch to closest human in group: {leader_ID}")
            #-------------------------------------------------------------------------

            self.following_id_vis = leader_ID
            self.previous_leader = leader_ID
            
            ###############
            # Set Subgoal #
            ###############
            '''
            - basis is human-robot vector
            - +/- 45 degree range
            - choose the one with highest min distance from human (the clearest path)
            '''
            vx = hr_vector[leader_ID][0]
            vy = hr_vector[leader_ID][1]
            
            base_gx = neighbor_traj[leader_ID][-1][0]
            base_gy = neighbor_traj[leader_ID][-1][1]

            def get_candidate_positions(vx, vy, angle_range=90): # 45
                positions = []
                magnitude = (vx**2 + vy**2)**0.5

                if magnitude != 0:
                    for angle in range(-angle_range, angle_range + 1, 2):
                        angle_rad = math.radians(angle)
                        rotated_vx = vx * math.cos(angle_rad) - vy * math.sin(angle_rad)
                        rotated_vy = vx * math.sin(angle_rad) + vy * math.cos(angle_rad)
                        pos_gx = base_gx - self.human_radius * rotated_vx / magnitude
                        pos_gy = base_gy - self.human_radius * rotated_vy / magnitude
                        positions.append((pos_gx, pos_gy))
                else:
                    positions.append((base_gx, base_gy))
                return positions

            def evaluate_position(goal_x, goal_y):
                min_distance = float('inf')
                for human in human_step:
                    if human[0] == leader_ID:
                        continue
                    human_px, human_py = human[1], human[2]
                    distance = math.sqrt((human_px - goal_x)**2 + (human_py - goal_y)**2)
                    min_distance = min(min_distance, distance)
                return min_distance    

            candidate_positions = get_candidate_positions(vx, vy)
            best_position = max(candidate_positions, key=lambda pos: evaluate_position(pos[0], pos[1]))
            new_gx, new_gy = best_position
            
            # if sample pos is close to neighbor, set further
            min_safe_distance = 2.0
            min_distance = evaluate_position(new_gx, new_gy)
            if min_distance < min_safe_distance:
                print('##### Push further ######')
                direction_vx = new_gx - base_gx
                direction_vy = new_gy - base_gy
                direction_mag = (direction_vx**2 + direction_vy**2)**0.5
                if direction_mag > 0:
                    scale = (min_safe_distance / min_distance) * 1.0
                    new_gx = base_gx + scale * direction_vx
                    new_gy = base_gy + scale * direction_vy
            
            # print(f"Selected position: ({new_gx:.2f}, {new_gy:.2f})")

            #################
            # Set New Speed #
            #################
            '''
            - move like leader when close
            - move fast to catch up leader when far
            '''
            command_v_pref=0
            if hr_distance[leader_ID] > self.catchup_threshold:
                command_v_pref = self.catchup_speed
                print("---catching up---")
            else:
                command_v_pref = human_speeds[leader_ID]

        else: # if no suitable leader, switch to default planner
            print(f"Back to default planner   distance: {goal_distance:.2f}")
            self.following_id_vis=-1
            new_gx = global_goal_x
            new_gy = global_goal_y
            command_v_pref = self.robot_speed_max

        #---------------------------------------------------------------------------------------

        if self.default:
            print("##### Using default SF planner #####")
            new_gx = global_goal_x
            new_gy = global_goal_y
            command_v_pref = self.robot_speed_max
        comand_goal=np.array([new_gx,new_gy])
        # print("Command Goal:", comand_goal)
        # print("Robot state:",state)
        action=self.base_controller.predict(comand_goal,state,human_step,mask,command_v_pref)

        try:
            self.state_buffer.pop(0)
            self.human_buffer.pop(0)
        except Exception as e:
            rospy.logwarn(f"Error while popping from buffers: {e}")

        return self.following_id_vis , action,comand_goal, invisible_list, visible_region_edges
    
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
        marker.header.frame_id = "robot_1/base_link"
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
