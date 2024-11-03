import base64
import json

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

        self.command_reasoning_publisher = self.create_publisher(
            String, 'command_reasoning',
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
            reasoning, command = self.get_command_from_openai(self.rgb_image, self.depth_image)
            if command:
                # Publish command
                command_msg = String()
                command_msg.data = command
                print(f"Publishing command: {command}")
                self.command_publisher.publish(command_msg)

                # Publish combined JSON as a string to `command_reasoning` topic
                command_reasoning_msg = String()
                command_reasoning_msg.data = json.dumps({"decision": command, "reasoning": reasoning})
                print(f"Publishing command with reasoning: {command_reasoning_msg.data}")
                print(command_reasoning_msg)
                self.command_reasoning_publisher.publish(command_reasoning_msg)

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
            Your job is to control a robot with the following instructions:
            - Move forward until you see a chair.
            - Turn left until you face a hallway, then stop.

            Choose the best decision based on the provided images and return a JSON response with exactly these two fields:
            {
                "decision": "turn_left/move_forward/turn_right/stop",
                "reasoning": "Explanation of the choice"
            }

            DO NOT RETURN ANYTHING ELSE. START WITH { AND END WITH }. DO NOT ADD UNNECESSARY ``` OR OTHER FORMATTING.
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
            response = client.chat.completions.create(
                model="gpt-4o",  # Adjust model as needed
                messages=messages,
                max_tokens=50,  # Adjust tokens for JSON output
                response_format={ "type": "json_object" }
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()
            parsed_response = json.loads(content)
            print(parsed_response)
            # Extract decision and reasoning
            decision = parsed_response.get("decision")
            reasoning = parsed_response.get("reasoning")
            
            # Ensure decision is valid
            if decision in ['turn_left', 'move_forward', 'turn_right', 'stop']:
                print(f"OpenAI Decision: {decision}, Reasoning: {reasoning}")
                return reasoning, decision
            else:
                self.get_logger().warning(f"Invalid decision received: {decision}")
                return None
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse JSON from OpenAI response: {e}")
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
