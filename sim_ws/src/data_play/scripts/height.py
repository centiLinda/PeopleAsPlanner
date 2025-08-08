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

# HEIGHT
from CrowdNav_HEIGHT.training.networks.model import Policy
from CrowdNav_HEIGHT.training.networks.envs import make_vec_envs
import CrowdNav_HEIGHT.crowd_sim # required
import importlib.util
import torch
import torch.nn as nn
import gym

class OurPlanner:
    def __init__(self,goal,obs_list):

        self.goal=goal
        self.obs_list=obs_list
        self.history_list=[]        
        self.list_length=25
        self.state_buffer=[] # list 25, FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta), latest at the end
        self.human_buffer=[] # list of list 25*n, list(ID, px, py, vx, vy), latest at the end
        self.time_step=0.11
        
        # HEIGHT   
        # config_path = '/root/sim_ws/src/data_play/scripts/CrowdNav_HEIGHT/trained_models/ours_RH_HH_hallwayEnv/configs/config.py'
        config_path = '/root/sim_ws/src/data_play/scripts/CrowdNav_HEIGHT/trained_models/train_from_scratch/configs/config.py'
        spec = importlib.util.spec_from_file_location("config", config_path)
        model_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_config)
        Config = getattr(model_config, 'Config')   
        env_config = config = Config()
        
        torch.manual_seed(config.env.seed)
        torch.cuda.manual_seed_all(config.env.seed)
        if config.training.cuda:
            if config.training.cuda_deterministic:
                # reproducible but slower
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            else:
                # not reproducible but faster
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
        torch.set_num_threads(1)
        device = torch.device("cuda" if config.training.cuda else "cpu")
        self.device = device
        
        # load_path = "/root/sim_ws/src/data_play/scripts/CrowdNav_HEIGHT/trained_models/ours_RH_HH_hallwayEnv/checkpoints/208200.pt"
        load_path = "/root/sim_ws/src/data_play/scripts/CrowdNav_HEIGHT/trained_models/train_from_scratch/checkpoints/10415_new.pt"
        env_name = config.env.env_name
        eval_dir = "."
        env_config.render_traj = False
        env_config.save_slides = False
        env_config.save_path = "."
        envs = make_vec_envs(env_name, config.env.seed, 1,
                            config.reward.gamma, eval_dir, device, allow_early_resets=True,
                            config=env_config, ax=None, test_case=-1)

        actor_critic = Policy(envs.observation_space.spaces,
                              envs.action_space,
                              base_kwargs=config,
                              base=config.robot.policy)
        actor_critic.load_state_dict(torch.load(load_path, map_location=device))
        actor_critic.base.nenv = 1
        nn.DataParallel(actor_critic).to(device)

        # evaluate()
        eval_recurrent_hidden_states = {}
        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['rnn'] = torch.zeros(1, 1, config.SRNN.human_node_rnn_size,
                                                          device=device)
        eval_masks = torch.zeros(1, 1, device=device)
        
        self.actor_critic = actor_critic
        self.eval_recurrent_hidden_states = eval_recurrent_hidden_states
        self.eval_masks = eval_masks        
        
        self.max_human_num = 20
        self.predict_steps = 5     
        self.v_max = 1.4 
        
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
                                        
        ob = {}            
        relative_goal_x = global_goal_x - state[0]
        relative_goal_y = global_goal_y - state[1]
        ob['robot_node'] = np.array([[0, 
                                      0, 
                                      relative_goal_x, 
                                      relative_goal_y, 
                                      np.arctan2(state[3], state[2])]]) # theta                                
        
        ob['temporal_edges']=np.array([[state[2], 
                                        state[3]]])   
        
        detectedHumanNum=min(len(human_step), self.max_human_num)
        current_human_states = np.ones((self.max_human_num, 4)) * np.inf # 15
        
        for i in range(detectedHumanNum):            
            traj = neighbor_traj[human_step[i][0]]
            # Compute current human state relative to the robot
            current_human_states[i, 0] = traj[-1][0] - state[0]  # x 
            current_human_states[i, 1] = traj[-1][1] - state[1]  # y
            current_human_states[i, 0] = traj[-1][2]  # vx
            current_human_states[i, 1] = traj[-1][3]  # vy
                        
        spatial_edges = current_human_states
            
        ob['spatial_edges'] = spatial_edges
        ob['spatial_edges'] = np.array(sorted(ob['spatial_edges'], key=lambda x: np.linalg.norm(x[:2])))
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15
        
        ob['detected_human_num'] = detectedHumanNum
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1
            
        # HEIGHT
        obs_num = 14
        scan = np.array(laser_scan) # 720
        scan[np.isinf(scan)] = 15
        scan[scan > 15] = 15
        ob['point_clouds'] = scan.reshape(180, 4).mean(axis=1)
        
        ob['obstacle_num'] = obs_num
        ob['obstacle_vertices'] = np.ones((max(1, obs_num), 8,)) * 15 # no need, only need point clouds
            
        obs = ob
        for key in obs:
            if isinstance(obs[key], np.ndarray):
                obs[key] = torch.from_numpy(obs[key][None, ...]).float().to(self.device) # add a dummy dimension
            else:
                obs[key] = torch.tensor(obs[key]).float().to(self.device)
                
        obs['point_clouds'] = obs['point_clouds'].unsqueeze(1)
                
        ###########
        # network #
        ###########
        with torch.no_grad():
            _, action, _, self.eval_recurrent_hidden_states = self.actor_critic.act(
                obs,
                self.eval_recurrent_hidden_states,
                self.eval_masks,
                deterministic=True)   
        
        # clip by max vel
        raw_action = action.cpu().numpy().flatten()
        act_norm = np.linalg.norm(raw_action)
        if act_norm > self.v_max:
            raw_action[0] = raw_action[0] / act_norm * self.v_max
            raw_action[1] = raw_action[1] / act_norm * self.v_max
        action = raw_action
        # print(f"-----------{action}")
                
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
        
        self.vx = 0
        self.vy = 0

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
            self.vx = action[0]
            self.vy = action[1]

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
            self.vx = 0
            self.vy = 0
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
            self.robot_state = [px, py, self.vx, self.vy]
        # print(f"robot state {vx},{vy}")
        
if __name__ == '__main__':
    try:
        robot = Robot()
        rospy.spin()  # Keep the node running, waiting for callbacks
    except rospy.ROSInterruptException:
        pass
