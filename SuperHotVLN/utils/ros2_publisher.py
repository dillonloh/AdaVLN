import cv2  

import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge  

class ROS2PublisherNode(Node):
    def __init__(self, super_hot_vln):
        super().__init__("ros2_publisher_node")
        self.bridge = CvBridge()

        # Set up QoS profile to keep only the last message
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,  # Only keep the last message
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL  # Ensures data is available to new subscribers
        )

        self.rgb_publisher = self.create_publisher(Image, "/rgb", qos_profile)
        self.depth_publisher = self.create_publisher(Image, "/depth", qos_profile)
        self.super_hot_vln = super_hot_vln

        # Subscribe to the /command topic
        self.command_subscription = self.create_subscription(
            String,
            "/command",
            self.command_callback,
            10
        )

    def command_callback(self, msg):
        # Update the _current_command and resume simulation
        print("Received command from agent:", msg.data)
        command = msg.data
        if command in ["turn_left", "turn_right", "move_forward", "stop"]:
            self.super_hot_vln._current_command = command
            self.super_hot_vln._world.play()  # Resume simulation if it's paused

    def publish_camera_data(self, rgb_data, depth_data):
        # Convert to numpy arrays to avoid any data persistence issues and ensure compatibility
        rgb_array = np.array(rgb_data, dtype=np.float32)
        depth_array = np.array(depth_data, dtype=np.float32)
        rgb_array = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Scale and convert RGB data to 8-bit format
        rgb_msg = self.bridge.cv2_to_imgmsg((rgb_array * 255).astype(np.uint8), encoding='rgb8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth_array, encoding='32FC1')

        # Publish the converted messages
        self.rgb_publisher.publish(rgb_msg)
        self.depth_publisher.publish(depth_msg)
