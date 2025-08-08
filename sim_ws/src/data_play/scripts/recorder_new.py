#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import glob
import time
from datetime import datetime
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Int32
import json
import numpy as np
import re

class SceneRecorder:
    def __init__(self):
        rospy.init_node("scene_recorder", anonymous=True)

        self.env_pub = rospy.Publisher("/env_control", Int32, queue_size=1)
        self.scene_name = rospy.get_param("scene", "nexus_2_0")
        self.scene_dir = os.path.join("/root/test_data", self.scene_name)  
        config_path = '/root/sim_ws/src/data_play/dataset/scene_config_30/' + self.scene_name + '.json'
        dt=1.0/30
        with open(config_path, 'r') as f:
            self.scene_config = json.load(f)  # Correct way to load JSON

        if not os.path.exists(self.scene_dir):
            rospy.logwarn("folder %s does not exit, creating...", self.scene_dir)
            os.makedirs(self.scene_dir)

        self.max_id = self.find_max_id()
        self.original_id = 0
        rospy.loginfo("max ID: %d", self.max_id)

        self.append_test_file(self.max_id)

        self.pub = rospy.Publisher("/scene_id", Int32, queue_size=1)
        self.pub.publish(self.max_id + 1)

        self.robot_position = None
        self.pos_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.robot_position_callback)

        # Create Trial Folder
        self.trial_folder = os.path.join(self.scene_dir, f"trial_{self.max_id}")
        os.makedirs(self.trial_folder, exist_ok=True)
        self.status_file=None
        self.finish_initial=False

        self.time_limit = (self.scene_config["end_frame"]-self.scene_config["start_frame"])/30

        self.running = True
        self.finished = True
        self.start_time = 0
        self.goal_range = 1.0
        self.reset_time = 0
        self.iteral_limit = 100

        self.scene_path = '/root/sim_ws/src/data_play/dataset/scene_config_30/' + self.scene_name + '.json'
        with open(self.scene_path, 'r') as f:
            self.scene_config = json.load(f)

        robot_pos_goal = np.array(self.scene_config["robot_start_end"])
        self.goal_pos = robot_pos_goal[2:4]

        self.timer = rospy.Timer(rospy.Duration(1.0 / 30), self.run)
        
        rospy.spin()

    def find_max_id(self):
        """ find max trial_{id} """
        trial_folders = glob.glob(os.path.join(self.scene_dir, "trial_*"))
        max_id = -1
        pattern = re.compile(r"trial_(\d+)")  # Match 'trial_{id}' and extract id

        for folder in trial_folders:
            folder_name = os.path.basename(folder)
            match = pattern.match(folder_name)
            if match:
                file_id = int(match.group(1))  # Extract numerical ID
                max_id = max(max_id, file_id)

        return max_id

    def append_test_file(self, max_id):
        """ `test.txt` add "time + id" """
        record_file = os.path.join(self.scene_dir, "record.txt")
        if not os.path.exists(record_file):
            rospy.logwarn("record.txt does not exist, creating...")
            open(record_file, "w").close()

        with open(record_file, "a") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{now} {max_id}\n")
        rospy.loginfo("written into record.txt: %s %d", now, max_id)

    def robot_position_callback(self, msg):
        """ track Gazebo all agent positions, save to individual actor files """
        if not self.running:
            return
        
        self.model_positions = {}  # Dictionary to store positions & rotations

        for i, model_name in enumerate(msg.name):
            position = msg.pose[i].position
            orientation = msg.pose[i].orientation

            # Store the position & rotation
            self.model_positions[model_name] = {
                "pos": (position.x, position.y, position.z),
                "rot": (orientation.x, orientation.y, orientation.z, orientation.w)
            }
        self.finish_initial=True

    def save_actor_states(self, timestamp):
        """ Saves all actor states at the given timestamp in separate files. """
        for model_name, data in self.model_positions.items():
            if model_name.startswith("actor_"):  # Only save actor models
                actor_id = model_name.split("_")[-1]  # Extract ID
                actor_file = os.path.join(self.trial_folder, f"actor_{actor_id}.txt")

                with open(actor_file, "a") as f:
                    pos_x, pos_y, pos_z = data["pos"]
                    rot_x, rot_y, rot_z, rot_w = data["rot"]
                    f.write(f"{timestamp} {pos_x} {pos_y} {pos_z} {rot_x} {rot_y} {rot_z} {rot_w}\n")

    def reset(self):
        """ Resets the scene and prepares for a new trial. """
        self.max_id += 1
        self.trial_folder = os.path.join(self.scene_dir, f"trial_{self.max_id}")
        os.makedirs(self.trial_folder, exist_ok=True)

        self.status_file = os.path.join(self.trial_folder, "status.txt")

        # Check if the file exists, create if necessary
        if not os.path.exists(self.status_file):
            open(self.status_file, "w").close()  # Create an empty file if it doesn't exist

        self.finished = False
        self.running = False
        self.robot_position = None

        msg = Int32()
        msg.data = 0
        self.env_pub.publish(msg)
        self.reset_time = rospy.Time.now().to_sec()
        try:
            self.model_positions.clear()
        except Exception as e:
            print(f"An error occurred: {e}")

        self.finish_initial=False
        print("Reset")

    def start(self):
        """ Starts the scene simulation. """
        msg = Int32()
        msg.data = 1
        self.env_pub.publish(msg)
        self.start_time = rospy.Time.now().to_sec()
        self.running = True
        print("Start")

    def run(self, event):
        """ Main loop that records actor states periodically. """
        if self.finished and self.running:
            self.reset()
            return 
        
        if not self.running and (rospy.Time.now().to_sec() - self.reset_time) > 3 and not self.finished:
            if self.max_id - self.original_id > self.iteral_limit:
                exit()
                rospy.signal_shutdown("Finished")
            self.start()
            return

        if self.finish_initial:
            try:
                timestamp = rospy.Time.now().to_sec() - self.start_time
                self.save_actor_states(timestamp)
                
                robot_pos = np.array([self.model_positions["robot_1"]["pos"][0], self.model_positions["robot_1"]["pos"][1]])

                with open(self.status_file, "a") as f:
                    # pos_x, pos_y, pos_z = data["pos"]
                    # rot_x, rot_y, rot_z, rot_w = data["rot"]
                    f.write(f"{timestamp} {robot_pos[0]} {robot_pos[1]}\n")
                
                if timestamp > self.time_limit:
                    with open(os.path.join(self.trial_folder, "status.txt"), "a") as f:
                        f.write("finished: out of time\n")
                    self.finished = True
                    print("out of time")
                    return

                if np.linalg.norm(robot_pos - self.goal_pos) < self.goal_range:
                    with open(os.path.join(self.trial_folder, "status.txt"), "a") as f:
                        f.write(f"success: {timestamp}\n")
                    self.finished = True
                    print("Finish")
                    return
            except Exception as e:
                print(f"An error occurred: {e}")
                return

    def __del__(self):
        """ close file """
        rospy.loginfo("Shutting down SceneRecorder.")

if __name__ == "__main__":
    try:
        SceneRecorder()
    except rospy.ROSInterruptException:
        pass
