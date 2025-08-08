#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Float32MultiArray, Int32MultiArray
from data_play.msg import ModelInfo

class ModelVisualizer:
    def __init__(self):
        rospy.init_node("model_visualization_node", anonymous=True)

        # Subscriber to ModelInfo topic
        self.model_sub = rospy.Subscriber("/gazebo/model_info", ModelInfo, self.model_callback)

        self.invisable_sub = rospy.Subscriber('/invisable_id', Int32MultiArray, self.invisable_callback)

        self.model_sub = rospy.Subscriber("/vis_array_topic", Float32MultiArray, self.array_callback)
        self.subgoal=[-1000.0,-1000.0]
        self.state=[-1002.0,-1003.0]
        self.leader_id=-1
        # Publisher for RViz markers
        self.marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)
        self.invisable_list=[]

        rospy.loginfo("Model Visualization Node Started")

    def invisable_callback(self, msg):
        self.invisable_list=msg.data

    def array_callback(self,msg):
        data=msg.data
        self.leader_id=int(data[0])
        self.subgoal=[data[1],data[2]]
        self.state=data[3:5]

    def create_ball_marker(self, position, marker_id):
        """
        Creates a sphere (ball) at the given position.
        """
        marker = Marker()
        marker.header.frame_id = "odom"  # Change according to your TF setup
        marker.header.stamp = rospy.Time.now()
        marker.ns = "balls"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5  # Ball size
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        if marker_id==self.leader_id:
            marker.color  = ColorRGBA(0, 0, 1.0, 1.0)
        elif marker_id in self.invisable_list:
            marker.color = ColorRGBA(1.0,1.0,0,1.0)
        else:
        # Set color (red for all)
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red, fully opaque

        return marker

    def create_sub_marker(self):
        """
        Creates a sphere (ball) at the given position.
        """
        marker = Marker()
        marker.header.frame_id = "odom"  # Change according to your TF setup
        marker.header.stamp = rospy.Time.now()
        marker.ns = "balls"
        marker.id = 10086
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x=self.subgoal[0]
        marker.pose.position.y=self.subgoal[1]
        marker.pose.position.z=0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5  # Ball size
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        marker.color  = ColorRGBA(0, 1.0, 0.0, 1.0)

        return marker

    def create_robot_marker(self):

        marker = Marker()
        marker.header.frame_id = "odom"  # Change according to your TF setup
        marker.header.stamp = rospy.Time.now()
        marker.ns = "cubes"
        marker.id = 10086
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x=self.state[0]
        marker.pose.position.y=self.state[1]
        marker.pose.position.z=0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5  # Ball size
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        marker.color  = ColorRGBA(0, 1.0, 0.0, 1.0)

        return marker

    def create_text_marker(self, position, marker_id, text):
        """
        Creates a text marker at the given position (slightly above the ball).
        """
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "labels"
        marker.id = marker_id + 1000  # Offset to avoid ID conflicts
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.position.z += 1.6  # Raise text above the ball
        marker.scale.z = 1.5  # Text size

        if marker_id==self.leader_id:
            marker.color  = ColorRGBA(1.0, 0, 0, 1.0)
        
        elif marker_id in self.invisable_list:
            marker.color = ColorRGBA(1.0,1.0,0,1.0)
        
        else:# Set text color (white)
            marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # White, fully opaque

        # Set the text
        marker.text = str(text)

        return marker

    def model_callback(self, msg):
        marker_array = MarkerArray()

        for i, pos in enumerate(msg.position):
            # Create a ball marker at each position
            ball_marker = self.create_ball_marker(pos, marker_id=msg.ids[i])
            marker_array.markers.append(ball_marker)

            # Create a text label with the ID
            text_marker = self.create_text_marker(pos, marker_id=msg.ids[i], text=msg.ids[i])
            marker_array.markers.append(text_marker)

        # Publish markers
        goal_marker=self.create_sub_marker()
        marker_array.markers.append(goal_marker)
        robot_marker=self.create_robot_marker()
        marker_array.markers.append(robot_marker)


        self.marker_pub.publish(marker_array)
        # rospy.loginfo(f"Published {len(marker_array.markers)} markers")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        visualizer = ModelVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass