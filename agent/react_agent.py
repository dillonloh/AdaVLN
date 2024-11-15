import base64
import json
import time
from datetime import datetime

from json_repair import repair_json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int16
import cv2
from cv_bridge import CvBridge
from openai import OpenAI
import os
import numpy as np
from dotenv import dotenv_values

config = dotenv_values(".env")  
client = OpenAI(api_key=config['OPENAI_API_KEY'])

class RGBDCommandNode(Node):
    def __init__(self, downscale_factor=0.5, history_limit=1):
        super().__init__('rgbd_command_node')
        self.downscale_factor = downscale_factor
        self.history_limit = history_limit
        self.episode_number = None
        self.step_count = 0  # To track each action step

        # Define QoS profile for latest message only
        qos_latest = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Initialize subscriptions for RGB, Depth images, Task Instructions, and Episode Number with latest-only QoS profile
        self.rgb_subscription = self.create_subscription(
            Image, '/rgb', self.rgb_callback, qos_profile=qos_latest
        )
        self.depth_subscription = self.create_subscription(
            Image, '/depth', self.depth_callback, qos_profile=qos_latest
        )
        self.task_instruction_subscription = self.create_subscription(
            String, '/current_task_instruction', self.task_instruction_callback, qos_profile=qos_latest
        )
        self.episode_subscription = self.create_subscription(
            Int16, '/episode_number', self.episode_callback, qos_profile=qos_latest
        )

        # Publisher for command and reasoning topics
        self.command_publisher = self.create_publisher(String, 'command', qos_profile=qos_latest)
        self.command_reasoning_publisher = self.create_publisher(String, 'command_reasoning', qos_profile=qos_latest)
        
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.task_instruction = None
        self.history = []  # Stores the history of reasoning and actions

        self.get_logger().info("RGBDCommandNode initialized and ready to receive images and process commands...")

    def episode_callback(self, msg):
        self.episode_number = msg.data
        self.get_logger().info(f"Episode number received: {self.episode_number}")
        self.setup_episode_directories()

    def setup_episode_directories(self):
        # Set up directory paths based on episode number
        self.run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = "results"
        self.dataset_path = os.path.join(self.results_dir, f"episode_{self.episode_number}_run_{self.run_start_time}.json")
        self.image_save_dir = os.path.join(self.results_dir, f"episode_{self.episode_number}_images")
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.get_logger().info(f"Directories set up for episode {self.episode_number}")

    def rgb_callback(self, msg):
        # self.get_logger().info("Received RGB image.")
        self.rgb_image = self.downscale_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
        # self.get_logger().info("RGB image downscaled.")
        self.process_images_if_ready()

    def depth_callback(self, msg):
        # self.get_logger().info("Received Depth image.")
        depth_image = self.downscale_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1'))
        
        # Set min and max depth values for visualization (similar to RViz2 settings)
        min_depth, max_depth = 0.0, 10.0
        cv_image_clipped = np.clip(depth_image, min_depth, max_depth)

        # Normalize to the 0-255 range for visualization as an 8-bit image
        cv_image_normalized = cv2.normalize(cv_image_clipped, None, 0, 255, cv2.NORM_MINMAX)
        cv_image_normalized = cv_image_normalized.astype('uint8')
        self.depth_image = cv_image_normalized

        # self.get_loger().info("Depth image downscaled.")
        self.process_images_if_ready()

    def task_instruction_callback(self, msg):
        self.get_logger().info(f"Received task instruction: {msg.data}")
        self.task_instruction = msg.data
        self.process_images_if_ready()

    def downscale_image(self, image):
        new_dimensions = (int(image.shape[1] * self.downscale_factor), int(image.shape[0] * self.downscale_factor))
        # self.get_logger().info(f"Downscaling image to dimensions: {new_dimensions}")
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    def save_image(self, image, image_type):
        # Use step count and timestamp to uniquely identify image files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.image_save_dir, f"{image_type}_{self.step_count}_{timestamp}.jpg")
        cv2.imwrite(filename, image)
        self.get_logger().info(f"Saved {image_type} image to {filename}")
        return filename
    def process_images_if_ready(self):
        if self.rgb_image is not None and self.depth_image is not None and self.task_instruction is not None:
            self.get_logger().info("All inputs received, beginning reasoning and action generation.")
            self.step_count += 1

            # Save RGB and Depth images with step count in filenames
            rgb_image_path = self.save_image(self.rgb_image, "rgb")
            depth_image_path = self.save_image(self.depth_image, "depth")
            
            decision, observation, obstacles, plan, reasoning = self.reason_and_act(
                self.rgb_image, self.depth_image, self.task_instruction
            )
            if decision:
                self.get_logger().info(f"Action generated: {decision}. Storing in history and publishing.")
                step_data = {
                    "episode": self.episode_number,
                    "step": self.step_count,
                    "task_instruction": self.task_instruction,
                    "rgb_image": rgb_image_path,
                    "depth_image": depth_image_path,
                    "decision": decision,
                    "observation": observation,
                    "obstacles": obstacles,
                    "plan": plan,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.history.append(step_data)
                
                # Write to JSON log file after each step
                self.write_step_to_json(step_data)

                # Publish action
                command_msg = String()
                command_msg.data = decision
                self.command_publisher.publish(command_msg)
                self.get_logger().info(f"Published action command: {decision}")

                command_reasoning_msg = String()
                command_reasoning_msg.data = json.dumps({
                    "decision": decision,
                    "observation": observation,
                    "obstacles": obstacles,
                    "plan": plan,
                    "reasoning": reasoning
                })
                self.command_reasoning_publisher.publish(command_reasoning_msg)
                self.get_logger().info("Published command with reasoning.")

                # If action is "stop", end the episode and reset task instruction
                if decision == "stop":
                    self.task_instruction = None

            # Reset images for the next cycle
            self.rgb_image = None
            self.depth_image = None
            self.get_logger().info("Resetting images for next cycle.")

    def encode_image_to_base64(self, image):
        self.get_logger().info("Encoding image to base64.")
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
        
    def reason_and_act(self, rgb_image, depth_image, task_instruction):
        base64_rgb = self.encode_image_to_base64(rgb_image)
        base64_depth = self.encode_image_to_base64(depth_image)
        self.get_logger().info("Base64 encoding of RGB and Depth images completed.")

        prompt = f'''

            You are to navigate and complete the task below.

            Your task is this:  {task_instruction}
            
            You are given egocentric RGB-D images with 60 degree FOV. These images are from the perspective of your eyes.
            Take note of this when navigating. Your end goals MUST be in the middle of your perspective and close to you. 

            You must avoid colliding with all walls and objects. You must not collide with any humans.

            Choose the best decision based on the provided images and return a JSON response with exactly these fields:
            {{
                "observation": "Description of what you see in the images.",
                "plan": "Description of your overarching plan to complete the task.",
                "reasoning": "Describe why you made this decision with reference to plan and observations.",
                "decision": "turn_left/move_forward/turn_right/stop"
            }}

            Here is an explanation of the different commands
            - turn_left: Rotate the robot to the left 30 degrees.
            - move_forward: Move the robot forward 1 meter.
            - turn_right: Rotate the robot to the right 30 degrees.
            - stop: Stop the robot and declare that you believe you have reached the destination. THIS CAN ONLY BE CALLED ONCE AND SHOULD ONLY BE CALLED
            WHEN YOU ARE DONE FOLLOWING ALL INSTRUCTIONS.

            MAKE SURE YOU ONLY STOP WHEN THE OBSERVATIONS IN THE IMAGE MATCH THE TASK INSTRUCTION'S END GOAL.
            DO NOT RETURN ANYTHING ELSE. START WITH {{ AND END WITH }}. DO NOT ADD UNNECESSARY ``` OR OTHER FORMATTING.

            You are also given the following truncated history of your decisions, HOWEVER YOU MIGHT BE
            HALLUCINATING, SO BE VERY CAREFUL ABOUT HOW YOU USE THIS HISTORY.
        '''
        
        recent_history = self.history[-self.history_limit:]
        self.get_logger().info(f"Using the latest {self.history_limit} entries from history.")
        
        history_text = "\n".join([
            f"Observation: {entry['observation']}\nObstacles: {entry['obstacles']}\nPlan: {entry['plan']}\nReasoning: {entry['reasoning']}\nAction: {entry['decision']}"
            for entry in recent_history
        ])
        full_prompt = prompt + "\n\n" + history_text

        messages = [
            {"role": "user", "content": full_prompt},
            {"role": "user", "content": f"data:image/jpeg;base64,{base64_rgb}"},
            # {"role": "user", "content": f"data:image/jpeg;base64,{base64_depth}"}
        ]

        retries = 5
        for attempt in range(retries):
            try:
                self.get_logger().info(f"Sending prompt to OpenAI API (Attempt {attempt + 1}/{retries}).")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )

                content = repair_json(response.choices[0].message.content.strip())
                self.get_logger().info(f"Received response from OpenAI API: {content}")
                parsed_response = json.loads(content)

                decision = parsed_response.get("decision")
                observation = parsed_response.get("observation")
                obstacles = parsed_response.get("obstacles")
                plan = parsed_response.get("plan")
                reasoning = parsed_response.get("reasoning")
                
                if decision in ['turn_left', 'move_forward', 'turn_right', 'stop']:
                    self.get_logger().info(f"Decision: {decision}, Observation: {observation}, Obstacles: {obstacles}, Plan: {plan}, Reasoning: {reasoning}")
                    return decision, observation, obstacles, plan, reasoning
                else:
                    self.get_logger().warning(f"Invalid decision received from OpenAI: {decision}")
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse JSON from OpenAI response: {e}")
            except Exception as e:
                self.get_logger().error(f"Error with OpenAI API: {e}")
            
            if attempt < retries - 1:
                self.get_logger().info(f"Retrying... (Attempt {attempt + 2}/{retries})")
                time.sleep(2)
        
        self.get_logger().error("Max retries reached. Returning None.")
        return None, None, None, None, None

    def write_step_to_json(self, step_data):
        # Append each step to the log file as it is generated
        with open(self.dataset_path, "a") as file:
            json.dump(step_data, file)
            file.write("\n")  # Add a newline to separate entries

    def save_run_history(self):
        self.get_logger().info(f"Final run history written to {self.dataset_path}")

def main(args=None):
    rclpy.init(args=args)
    node = RGBDCommandNode(downscale_factor=0.5, history_limit=1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RGBD Command Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
