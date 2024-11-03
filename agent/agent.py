import base64

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
import cv2
from cv_bridge import CvBridge
from openai import OpenAI
import os
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")  
print(config)
client = OpenAI(api_key=config['OPENAI_API_KEY'])


# Set OpenAI API Key from environment variable

class RGBDCommandNode(Node):
    def __init__(self):
        super().__init__('rgbd_command_node')

        # Initialize subscriptions for RGB and Depth images (non-persistent)
        self.rgb_subscription = self.create_subscription(
            Image, '/rgb', self.rgb_callback, qos_profile=rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.VOLATILE)
        )

        self.depth_subscription = self.create_subscription(
            Image, '/depth', self.depth_callback, qos_profile=rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.VOLATILE)
        )

        # Subscription for the ready_for_command signal (non-persistent)
        self.ready_subscription = self.create_subscription(
            Bool, '/ready_for_command', self.ready_callback,
            qos_profile=rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.VOLATILE)
        )

        # Publisher for the command topic (non-persistent)
        self.command_publisher = self.create_publisher(
            String, 'command',
            qos_profile=rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.VOLATILE)
        )

        # OpenCV bridge for ROS image conversion
        self.bridge = CvBridge()

        # Placeholders for images and readiness flag
        self.rgb_image = None
        self.depth_image = None
        self.ready_for_command = False
        print("Ready to receive images and process commands...")
    def rgb_callback(self, msg):
        # Store the received RGB image
        print("Received RGB image")
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.ready_for_command = msg.data
        self.process_images_if_ready()

    def depth_callback(self, msg):
        # Store the received Depth image
        print("Received Depth image")
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_images_if_ready()

    def ready_callback(self, msg):
        # Update readiness based on the received Bool message
        print(f"Received ready_for_command: {msg.data}")
        self.ready_for_command = msg.data
        self.process_images_if_ready()

    def process_images_if_ready(self):
        # Process images if both are available and ready_for_command is True
        print("Processing images...")
        if self.ready_for_command and self.rgb_image is not None and self.depth_image is not None:
            # Process images and get command
            command = self.get_command_from_openai(self.rgb_image, self.depth_image)
            if command:
                # Publish command
                command_msg = String()
                command_msg.data = command
                print(f"Publishing command: {command}")
                self.command_publisher.publish(command_msg)

            # Clear images and reset readiness flag after processing
            self.rgb_image = None
            self.depth_image = None
            self.ready_for_command = False

    def encode_image_to_base64(self, image):
        # Encode OpenCV image to base64
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def get_command_from_openai(self, rgb_image, depth_image):
        # Convert images to base64 for API usage
        base64_rgb = self.encode_image_to_base64(rgb_image)
        base64_depth = self.encode_image_to_base64(depth_image)

        # Prepare OpenAI API call with the images and prompt
        prompt = '''
            Your current job is to move forward until you see a chair. Then, turn left until you face a hallway, then stop.
            You must choose from one of the following options, and ONLY return the chosen option with NO OTHER text.
            (Options: turn_left, move_forward, turn_right, stop)

        '''
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_rgb}",
                            "detail": "auto"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_depth}",
                            "detail": "auto"
                        },
                    },
                ],
            }
        ]

        try:
            # Call the OpenAI API
            response = client.chat.completions.create(model="gpt-4o",  # Adjust if needed
            messages=messages,
            max_tokens=10)
            print(f"OpenAI response: {response.choices[0].message.content}")
            command = response.choices[0].message.content.strip()
            if command in ['turn_left', 'move_forward', 'turn_right']:
                return command
            else:
                self.get_logger().warning(f"Invalid command received: {command}")
                return None
        except Exception as e:
            self.get_logger().error(f"Error with OpenAI API: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = RGBDCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RGBD Command Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
