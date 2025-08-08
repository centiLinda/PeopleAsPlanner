#!/usr/bin/env python3

import os
import numpy as np
import glob
import re
import json

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate, translate
from matplotlib.animation import FuncAnimation

class obstacle:
    def __init__(self, vertices):
        """
        Initialize an obstacle as a polygon.
        
        :param vertices: List of (x, y) tuples representing the polygon vertices.
        """
        self.polygon = Polygon(vertices)
        self.robot_radius=0.35

    def collsion(self, point):
        """
        Calculate the shortest distance from a given point to the obstacle.
        
        :param point: (x, y) tuple representing the point.
        :return: Shortest distance to the polygon.
        """
        p = Point(point)
        if p.distance(self.polygon)<self.robot_radius:
            return 1
        else:
            return 0

    def draw(self):
        """
        Draw the obstacle with a black fill.
        
        :param ax: Matplotlib axis to draw on.
        """
        x, y = self.polygon.exterior.xy
        ax.fill(x, y, color='black', alpha=1)  # Fill the polygon in black

class agent:
    def __init__(self,id,data,track_dict):
        self.robot_radius=0.5
        self.human_radius=0.5
        self.id = id
        if id <0:
            self.type=0
            return
        self.data=data
        # print(track_dict)
        if track_dict[id]=="Car" or track_dict[id]=="Bus":
            self.type=2
        elif track_dict[id]=="Biker" or track_dict[id]=="Cart":
            self.type=1
        else:
            self.type=0

    
    def _create_footprint(self, px, py, yaw):
        """Create footprint shape based on type, position, and yaw"""
        if self.type == 0:
            return Point(px, py).buffer(self.human_radius)  # Circle with radius 0.5

        elif self.type == 1:
            rect = Polygon([(-0.9, -0.5), (0.9, -0.5), (0.9, 0.5), (-0.9, 0.5)])  # 2x1 rectangle

        elif self.type == 2:
            rect = Polygon([(-2.25, -0.95), (2.25, -0.95), (2.25, 0.95), (-2.25, 0.95)])  # 4x2 rectangle
        else:
            raise ValueError("Invalid agent type")

        # Rotate and move to the specified position
        rotated_rect = rotate(rect, np.degrees(yaw), origin=(0, 0), use_radians=False)
        return translate(rotated_rect, xoff=px, yoff=py)

    def cal_dist(self, px, py, yaw, target_point):
        """
        Calculate the shortest distance from a target point to the agent's footprint
        :param px: New x-coordinate of agent's center
        :param py: New y-coordinate of agent's center
        :param yaw: Rotation angle in radians
        :param target_point: (x, y) coordinates of the target point
        :return: Shortest distance to the footprint
        """
        footprint = self._create_footprint(px, py, yaw)
        point = Point(target_point)
        return point.distance(footprint)
    
    def cal_index_collision(self,index,target_point):

        px=self.data[index,1]
        py=self.data[index,2]
        yaw=self.data[index,3]
        dist=self.cal_dist(px,py,yaw,target_point)
        if dist<=self.robot_radius:
            return 1
        else:
            return 0

    def draw_agent(self, px, py, yaw):
        """Draw the agent's footprint"""
        footprint = self._create_footprint(px, py, yaw)
        if self.type == 0:
            # Draw circle
            color='blue'
            if self.id>0:
                circle = plt.Circle((px, py), self.human_radius, color='blue', alpha=0.5)
            else:
                circle = plt.Circle((px, py), self.robot_radius, color='red', alpha=0.5)

            ax.add_patch(circle)
        else:
            # Draw rectangle
            x, y = footprint.exterior.xy
            ax.fill(x, y, color='blue', alpha=0.5)
        # Draw agent center
        ax.scatter(px, py, color='black', marker="o", s=50)
    
    def draw_agent_index(self,index):
        px,py,yaw=self.data[index,1:4]
        """Draw the agent's footprint"""
        footprint = self._create_footprint(px, py, yaw)
        
        if self.type == 0:
            # Draw circle without the center dot
            circle = plt.Circle((px, py), self.human_radius, color='blue', alpha=0.5)
            ax.add_patch(circle)

            ax.text(px, py, str(self.id), color='white', fontsize=8, ha='center', va='center', fontweight='bold')

        else:
            # Draw rectangle
            x, y = footprint.exterior.xy
            ax.fill(x, y, color='blue', alpha=0.5)

            ax.text(px, py, str(self.id), color='white', fontsize=8, ha='center', va='center', fontweight='bold')

def quaternion_to_yaw(qx, qy, qz, qw):
    """
    Convert quaternion (qx, qy, qz, qw) to yaw angle (rotation around Z-axis).
    Formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    """
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return yaw

def read_actor_data(trial_folder):
    """
    Reads actor_{id}.txt files in the given trial folder.
    Extracts time, px, py, pz and calculates yaw from quaternion.
    Returns a dictionary where keys are actor IDs and values are NumPy arrays.
    """
    actor_files = glob.glob(os.path.join(trial_folder, "actor_*.txt"))
    actor_data = {}

    for file in actor_files:
        actor_id = int(re.search(r"actor_(\d+).txt", file).group(1))  # Extract actor ID
        data = np.loadtxt(file)  # Load numerical data

        if data.ndim == 1:  # If only one row exists, reshape to avoid indexing issues
            data = data.reshape(1, -1)

        time, px, py, pz, qx, qy, qz, qw = data.T  # Transpose to extract columns

        # Compute yaw angle
        yaw_angles = np.array([quaternion_to_yaw(qx[i], qy[i], qz[i], qw[i]) for i in range(len(qx))])

        # Store in dictionary
        actor_data[actor_id] = np.column_stack((time, px, py, yaw_angles))  # Combine with yaw

    return actor_data

def read_status_data(trial_folder):
    """
    Reads status.txt, extracts time, px, py into a NumPy array,
    and retrieves the last row's status message.
    """
    status_file = os.path.join(trial_folder, "status.txt")
    print(status_file)
    if not os.path.exists(status_file):
        raise FileNotFoundError(f"Status file not found: {status_file}")

    # Read all lines
    with open(status_file, "r") as f:
        lines = f.readlines()

    # Extract numerical data
    status_data = []
    for line in lines[:-1]:  # Ignore last line for now
        parts = line.strip().split()
        if len(parts) == 3:  # Ensure format is time px py
            status_data.append([float(parts[0]), float(parts[1]), float(parts[2])])

    # Convert to NumPy array
    status_data = np.array(status_data) if status_data else np.empty((0, 3))

    # Extract last line message
    last_line = lines[-1].strip()
    print(trial_folder)
    finish_time = None
    if last_line.startswith("success:"):
        finish_time = float(last_line.split(":")[1].strip())  # Extract finish time

    return status_data, last_line, finish_time

def process_scene(scene_name):
    """
    Given a scene name, processes all trial data.
    """
    base_path = f"/root/test_data/{scene_name}/"
    trial_folders = glob.glob(os.path.join(base_path, "trial_*"))

    results = {}

    for trial_folder in trial_folders:


        trial_id = os.path.basename(trial_folder).split("_")[-1]  # Extract trial ID
        print(trial_id)
        if int(trial_id)<0:
            continue

        # Read actor data
        actor_data = read_actor_data(trial_folder)

        # Read status data
        status_data, last_status_message, finish_time = read_status_data(trial_folder)

        # Store in results
        results[trial_id] = {
            "actor_data": actor_data,  # Dictionary of actor_{id}: numpy array
            "status_data": status_data,  # NumPy array of time px py
            "last_status_message": last_status_message,  # String message
            "finish_time": finish_time,  # Finish time if success
        }

    return results

# Example usage:
scene_name = "crossing_0"
# scene_name = "promenade_0"
# scene_name = "roundabout_0"
scene_results = process_scene(scene_name)

track_dict={}
with open(f"/root/sim_ws/src/data_play/temp/{scene_name}/pair/data_id_label_pairs.txt", "r") as file:
    next(file)  # Skip the first line
    for line in file:
        parts = line.strip().split(" ", 1)  # Split at the first space only
        if len(parts) == 2:
            track_id, label = parts
            track_dict[int(track_id)] = label  # Convert track_id to int for consistency
print(track_dict)

config_path = '/root/sim_ws/src/data_play/dataset/scene_config_30/' + scene_name + '.json'
with open(config_path, 'r') as f:
    scene_config = json.load(f)  # Correct way to load JSON

obs_list=[]
for obs in scene_config["obstacles"]:
    obs_list.append(obstacle(np.array(obs)))

data_vis=True

file_path =f"/root/test_data/{scene_name}/experiment_log.csv"

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Deleted existing file: {file_path}")

with open(file_path, "w") as file:
    file.write("Trial_ID, Finish, Total_collision, traveling_dis, Total_Time\n")  # Header
    file.close()
    # Print the results
for trial_id, data in scene_results.items():
    print(f"\nTrial {trial_id}:")
    print("Last Status Message:", data["last_status_message"])
    finish_flag=0

    if data["finish_time"] is not None:
        print("Finish Time:", data["finish_time"])
        finish_flag=1
    else:
        print("Out of Time:")
    
    total_time=data["finish_time"]


    status=data["status_data"]

    agent_list=[]
    for actor_id, actor_array in data["actor_data"].items():
        agent_list.append(agent(actor_id,actor_array,track_dict))

    if data_vis:
    # Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Agent Visualization")
        plt.ion()
    human_collision=0
    obs_collision=0
    robot_draw=agent(-1,[],[])
    # For now agent_list has the agent so we need to visulize them

    traveling_dis=0
    previous_position=None
    for index,row in enumerate(status):
        robot_position=row[1:4]
        # print(robot_position)

        if previous_position is None:
            previous_position=robot_position

        traveling_dis=traveling_dis+np.linalg.norm(robot_position-previous_position)

        previous_position=robot_position

        for ped in agent_list:
            if ped.cal_index_collision(index,robot_position)==1:
                human_collision=human_collision+1
        
        for obs in obs_list:
            if obs.collsion(robot_position):
                obs_collision=obs_collision+1
                
        if data_vis:
            ax.clear()

            ax.set_xlim(robot_position[0] - 10, robot_position[0] + 10)
            ax.set_ylim(robot_position[1] - 10, robot_position[1] + 10)
            # Draw collision counts in the top-right corner
            ax.text(0.95, 0.95, f"Human Collisions: {human_collision}", 
                    transform=ax.transAxes, fontsize=12, color='red', ha='right', va='top', fontweight='bold')

            ax.text(0.95, 0.90, f"Obstacle Collisions: {obs_collision}", 
                    transform=ax.transAxes, fontsize=12, color='blue', ha='right', va='top', fontweight='bold')
            ax.text(0.95, 0.85, f"Frame Index: {index}", 
                    transform=ax.transAxes, fontsize=12, color='blue', ha='right', va='top', fontweight='bold')
            ax.text(0.95, 0.80, f"Traveling Dis: {traveling_dis}", 
                    transform=ax.transAxes, fontsize=12, color='blue', ha='right', va='top', fontweight='bold')

        if data_vis:
            for obs in obs_list:
                obs.draw()
            for ped in agent_list:
                ped.draw_agent_index(index)

            robot_draw.draw_agent(robot_position[0],robot_position[1],0)
            plt.pause(0.03)
        

        # fig.clf()  
    if data_vis:     
        ax.legend()
        ax.grid(True)
        plt.show()
    
    total_collision=human_collision+obs_collision
    with open(file_path, "a") as file:
        file.write(f"{trial_id},{finish_flag},{total_collision},{traveling_dis},{total_time}\n")
        file.close()
