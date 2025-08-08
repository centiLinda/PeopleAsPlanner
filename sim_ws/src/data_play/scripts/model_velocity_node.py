#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from data_play.msg import ModelInfo
from geometry_msgs.msg import Point
import numpy as np
import re

class GazeboModelVelocity:
    def __init__(self):
        rospy.init_node('model_velocity_node', anonymous=True)
        self.model_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        self.vel_pub = rospy.Publisher('/gazebo/model_info', ModelInfo, queue_size=1)
        self.previous_states = {}  # Store previous positions and timestamps
        self.rate = rospy.Rate(10)  # 10Hz publishing rate
        self.actor_pattern = re.compile(r'actor_\d+')  # Regex pattern to match actor IDs
        self.freq = 10
    
    def model_states_callback(self, msg):
        model_positions = []
        model_velocities = []
        model_ids = []
        model_tags = []
        existing_models = set(msg.name)
        
        for i, model_name in enumerate(msg.name):
            if self.actor_pattern.fullmatch(model_name):
                model_id = int(model_name.split('_')[1])  # Extract numeric ID
                position = msg.pose[i].position
                
                # Determine tag based on the existence of corresponding bike_{id} or car_{id}
                tag = 1 if f'bike_{model_id}' in existing_models else 2 if f'car_{model_id}' in existing_models else 0
                
                # Compute velocity if previous state exists
                if model_id in self.previous_states:
                    prev_position, prev_time = self.previous_states[model_id]
                    current_time = rospy.Time.now().to_sec()
                    dt = current_time - prev_time
                    if dt < 1.0 / self.freq:
                        return
                    if dt > 0:
                        velocity = Point(
                            (position.x - prev_position.x) / dt,
                            (position.y - prev_position.y) / dt,
                            (position.z - prev_position.z) / dt
                        )
                        # Check if velocity magnitude exceeds threshold
                        norm_velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                        if norm_velocity > 100:
                            velocity = Point(0.0, 0.0, 0.0)
                        

                    else:
                        velocity = Point(0.0, 0.0, 0.0)
                else:
                    velocity = Point(0.0, 0.0, 0.0)
                
                # Update stored states
                self.previous_states[model_id] = (position, rospy.Time.now().to_sec())
                
                # Append data to message fields
                model_positions.append(position)
                model_velocities.append(velocity)
                model_ids.append(model_id)
                model_tags.append(tag)
        
        # Publish model info
        if model_positions:
            model_info_msg = ModelInfo(
                position=model_positions,
                velocity=model_velocities,
                ids=model_ids,
                tags=model_tags
            )
            self.vel_pub.publish(model_info_msg)
    
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    try:
        node = GazeboModelVelocity()
        node.run()
    except rospy.ROSInterruptException:
        pass
