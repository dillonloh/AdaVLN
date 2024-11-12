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
        self.task_instruction_publisher = self.create_publisher(String, "/current_task_instruction", qos_profile)
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

        try:
            # Scale and convert RGB data to 8-bit format
            rgb_data_bgra = cv2.cvtColor(rgb_data, cv2.COLOR_RGBA2BGRA)
            
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_data_bgra, encoding='bgra8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth_data, encoding='32FC1')

            # Publish the converted messages
            self.rgb_publisher.publish(rgb_msg)
            self.depth_publisher.publish(depth_msg)
        except Exception as e:
            print("Error while publishing camera data:", e)
            
    def publish_current_task_instruction(self, task_instruction):
        task_instruction_msg = String()
        task_instruction_msg.data = task_instruction
        self.task_instruction_publisher.publish(task_instruction_msg)