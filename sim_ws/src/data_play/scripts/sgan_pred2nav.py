#!/usr/bin/env python3

import rospy
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
import os

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

# Pred2Nav
from Pred2Nav.crowd_nav.policy.vecMPC.predictors import *
from Pred2Nav.crowd_nav.configs.config_vecmpc import Config

class OurPlanner:
    def __init__(self,goal,obs_list):

        self.goal=goal
        self.obs_list=obs_list
        self.history_list=[]        
        self.list_length=25
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end
        self.time_step=0.11
        
        # Pred2Nav
        self.span = np.deg2rad(360)
        self.n_actions = 100
        self.vpref = 1.4
        self.rollout_steps = 6 # mpc_rollout_steps
        self.prediction_horizon = 6 # predictor horizon
        self.prediction_dt = 0.1
        
        # c = 'cv'
        # self.predictor = CV()
        c = 'sgan'
        self.predictor = SGAN()
        
        config = Config(c)
        config.save_path = os.path.join('Pred2Nav_output')
        config.exp_name = c
        self.predictor.vpref = self.vpref
        self.predictor.dt = self.prediction_dt
        self.predictor.set_params(config.MPC['params'])
        
    def generate_cv_action(self, position, goal, vpref):
        """To get particular action"""
        dxdy = goal-position # (2, N)
        thetas = np.arctan2(dxdy[1], dxdy[0]) # (N, )
        return np.stack((np.cos(thetas), np.sin(thetas)), axis=0) * vpref # (2, N)
    
    def integrator(self, S, U):
        M = 4
        dt_ = float(self.predictor.dt) / M
        S_next = np.array(S)
        for i in range(M):
            k1 = dt_ * self.state_dot(S, U)
            k2 = dt_ * self.state_dot(S + (0.5 * k1), U)
            k3 = dt_ * self.state_dot(S + (0.5 * k2), U)
            k4 = dt_ * self.state_dot(S + k3, U)
            S_next += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return S_next

    @staticmethod
    def state_dot(S0, U):
        S_dot = np.array(S0)
        S_dot[0] = S0[1]
        S_dot[1] = ((-38.73 * S0[0]) + (-11.84 * S0[1]) + (-6.28 * S0[2]) +
                    (51.61 * U[0]) + (11.84 * U[1]) + (6.28 * U[2]))

        S_dot[2] = ((13.92 * S0[0]) + (2.0 * S0[1]) + (1.06 * S0[2]) +
                    (-8.72 * U[0]) + (-2.0 * U[1]) + (-1.06 * U[2]))

        S_dot[3] = S0[4]
        S_dot[4] = ((-38.54 * S0[3]) + (-11.82 * S0[4]) + (-6.24 * S0[5]) +
                    (51.36 * U[3]) + (11.82 * U[4]) + (6.24 * U[5]))

        S_dot[5] = ((14.00 * S0[3]) + (2.03 * S0[4]) + (1.07 * S0[5]) +
                    (-8.81 * U[3]) + (-2.03 * U[4]) + (-1.07 * U[5]))
        return S_dot
    
    def step_dynamics(self, position, state, action): # (2, N), (6, N), (2, N)
        # Reference input (global)
        N = action.shape[1]
        U = np.concatenate((np.zeros((2, N)), action[0, None], np.zeros((2, N)), action[1, None]), axis=0) # (6, N)

        # Integrate with ballbot dynamics
        next_state = self.integrator(state, U) # (6, N)

        velocity = next_state[[2, 5]] # (2, N)
        position = position +  velocity * self.prediction_dt # (2, N)
        return next_state, position, velocity
        
    def generate_action_set(self, pos, vel, theta, theta_dot, goal, sim_heading):
        '''
        pos, vel, theta, theta_dot, goal: np.array [2]
        sim_heading: np.float64
        '''
        thetas = [sim_heading-(self.span / 2.0) + i * self.span / (self.n_actions - 1) for i in range(self.n_actions)]
        thetas = thetas if len(thetas) > 1 else np.arctan2(goal-pos)
        goals = pos[:, None] + (self.vpref * self.prediction_horizon *10) * np.stack((np.cos(thetas), np.sin(thetas)), axis=0) # (2, N)
        state = np.array([theta[0], theta_dot[0], vel[0], theta[1], theta_dot[1], vel[1]]) # (6, )

        rollouts = []
        state = np.repeat(state[:, None], goals.shape[1], axis=1)  # (6, N)
        pos = np.repeat(pos[:, None], goals.shape[1], axis=1)  # (2, N)
        for _ in range(self.rollout_steps):
            ref_velocities = self.generate_cv_action(pos, goals, self.vpref) # (2, N)
            state, pos, _ = self.step_dynamics(pos, state, ref_velocities)
            rollouts.append( np.concatenate((pos, ref_velocities), axis=0).transpose((1, 0)))
        rollouts = np.stack(rollouts, axis=1) # N x T x 4
        return rollouts # np.array (n_actions, rollout_steps, [px, py, vx, vy]) 10, 6, 4
    
    def action_post_processing(self, action):
        action_xy = np.array([action[0], action[1]])

        if np.linalg.norm(action) > 0:
            self.sim_heading = np.arctan2(action[1], action[0])
        else:
            # Randomly set heading to get us unstuck if not moving
            if np.linalg.norm(self.pathbot_state.get_velocity()) < 0.1:
                self.sim_heading = np.random.rand() * 2 * np.pi
        
        return action_xy

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
        
        neighbor_traj = {}
        for timestep in self.human_buffer:
            for human in timestep:
                id, px, py, vx, vy = human
                if any(agent[0] == id for agent in human_step): # only add visible human
                    if id not in neighbor_traj:
                        neighbor_traj[id] = []
                    neighbor_traj[id].append([px, py, vx, vy])
        
        '''
        pos, vel, theta, theta_dot, goal: np.array [2]
        sim_heading: np.float64
        trajectory: np.array (t, n_agent, 5_states) new state is append to the end
        all_state: np.array [px, py, vx, vy, radius] robot, human1, human2, ...
        '''
        radius = 0.5
        goal = np.array(self.goal)
        pos = np.array([state[0], state[1]])
        vel = np.array([state[2], state[3]])
        sim_heading = np.arctan2(vel[1], vel[0]) # vy, vx
        theta = np.array([0, 0])        # dummy
        theta_dot = np.array([0, 0])    # dummy
        trajectory = []
        all_state = []
                            
        for human_timestep, robot_timestep in zip(self.human_buffer, self.state_buffer):
            timestep_state = []

            robot_state = np.array(robot_timestep)
            robot_state = np.append(robot_state, radius)
            timestep_state.append(robot_state)
            
            for human in human_timestep:
                id, px, py, vx, vy = human
                if any(agent[0] == id for agent in human_step): # only add visible human
                    human_state_with_radius = np.array([px, py, vx, vy])
                    human_state_with_radius = np.append(human_state_with_radius, radius)
                    timestep_state.append(human_state_with_radius)
                    
            trajectory.append(timestep_state)
        
        # pad to the same length (if new human occurs)
        empty_state = np.array([0, 0, 0, 0, 0.1])
        max_length = max(len(item) for item in trajectory)
        padded_trajectory = []
        for item in trajectory:
            padded_item = item.copy()
            while len(padded_item) < max_length:
                padded_item.append(empty_state)
            padded_trajectory.append(padded_item)
        padded_trajectory = np.array(padded_trajectory)    
             
        trajectory = padded_trajectory
        all_state = trajectory[-1]
        
        action_set = self.generate_action_set(pos, vel, theta, theta_dot, goal, sim_heading)
        predictions, costs, action_set, best_action = self.predictor.predict(trajectory, all_state, action_set, goal)
        action = self.action_post_processing(best_action[2:])
        # print(f"--------------------Action: {action}")
        
        try:
            self.state_buffer.pop(0)
            self.human_buffer.pop(0)
        except Exception as e:
            rospy.logwarn(f"Error while popping from buffers: {e}")

        return 0, action, [0, 0], [0]
    
    def clear_buffer(self):
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end


class Robot:

    def __init__(self):
        self.lock=threading.Lock()
        rospy.init_node('robot_listener', anonymous=True)
        scene=rospy.get_param("scene")
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

    def laser_callback(self, msg):
        self.laser_scan = msg.ranges
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
            track_id,action,subgoal,invis_index=self.planner.predict(state_now,model_temp,self.id_mask,self.laser_scan) #NOTE
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
            # self.invisable_pub.publish(invis_id_msg)

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
