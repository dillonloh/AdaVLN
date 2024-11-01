# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import random
import time

import cv2  
import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController, WheelBasePoseController
from omni.isaac.core.objects import VisualCuboid
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import carb

    
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
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
        if command in ["turn_left", "turn_right", "move_forward"]:
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

class SuperHotVLN(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._moving_objects = []
        if not rclpy.ok():
            rclpy.init()
        self.ros2_node = ROS2PublisherNode(self)  # Initialize ROS2 node with self reference

        # Create and start the ROS 2 executor on a separate thread
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.ros2_node)
        self.ros2_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros2_thread.start()

        return

    def setup_scene(self):

        world = self.get_world()

        matterport_env_usd = "/home/dillon/0Research/IsaacSim/test.usd"
        matterport_env_prim_path = "/World"

        add_reference_to_stage(usd_path=matterport_env_usd, prim_path=matterport_env_prim_path)
        assets_root_path = get_assets_root_path()

        # jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot_prim_path = "/World/Jetbot"
        
        # Add the Jetbot camera
        camera_prim_path = "/World/Jetbot/chassis/rgb_camera/jetbot_camera"

        world.scene.add(
            Camera(
                prim_path=camera_prim_path,
                name="jetbot_camera",
                resolution=(1280, 720),
            )
        )

        # Add the Jetbot robot
        world.scene.add(
            WheeledRobot(
                prim_path=jetbot_prim_path,
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                name="jetbot", 
                position=[0.0, 0.0, 0.0]
            )
        )

        # Add random moving objects (e.g., cubes)
        for i in range(5):  # Add 5 random objects
            random_position = [random.uniform(-5, 5), random.uniform(-5, 5), 0.5]
            random_object_prim_path = f"/World/RandomObject_{i}"
            world.scene.add(
                VisualCuboid(
                    prim_path=random_object_prim_path,
                    name=f"random_object_{i}",
                    position=random_position,
                    scale=np.array([0.5015, 0.5015, 0.5015]), # most arguments accept mainly numpy arrays.
                    color=np.array([0, 0, 1.0]), # RGB channels, going from 0-1
                )
            )
            self._moving_objects.append(world.scene.get_object(f"random_object_{i}"))

        return

    async def setup_post_load(self):
        self._start_time = time.time()
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("jetbot")


        self._world.add_physics_callback("moving_objects", callback_fn=self.move_objects_in_random_paths)
        # self._world.add_physics_callback("publish_camera_data", callback_fn=self.publish_camera_data)
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        self._jetbot_controller = DifferentialController(name="jetbot_control", wheel_radius=0.035, wheel_base=0.1)
        self._current_command = None
        self._target_position = None
        self._target_yaw = None
        self._camera = self._world.scene.get_object("jetbot_camera")   
        
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()
        self._camera.add_distance_to_image_plane_to_frame()
        self.bridge = CvBridge()
        

    def publish_camera_data(self):
        print("Publishing camera data...")
        # Capture RGB and Depth images from the camera
        rgb_data = self._camera.get_rgb()
        depth_data = self._camera.get_depth()
        # print(f"Type of RGB data: {type(rgb_data)} | Type of Depth data: {type(depth_data)}")
        # print(f"RGB data shape: {rgb_data.shape} | Depth data shape: {depth_data.shape}")
        # print(f"RGB data type: {rgb_data.dtype} | Depth data type: {depth_data.dtype}")
        # print(f"RGB data range: {rgb_data}")
        # Publish the data using ROS2PublisherNode
        self.ros2_node.publish_camera_data(rgb_data, depth_data)

    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def move_objects_in_random_paths(self, step_size):
        print("Moving objects in random paths...")
        # Move each object in a random direction with a small step size
        for i, obj in enumerate(self._moving_objects):
            # Generate a random direction and move the object slightly
            random_direction = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0])
            current_position, current_orientation = obj.get_world_pose()
            new_position = np.array(current_position) + random_direction
            print(f"Object {i+1} current position: {current_position}")
            obj.set_world_pose(new_position.tolist(), current_orientation)

    def send_robot_actions(self, step_size):
        print("Sending robot actions...")
        # Get the current position and orientation (in quaternion) of the robot
        position, orientation_quat = self._jetbot.get_world_pose()

        # Convert quaternion (QW, QX, QY, QZ) to Euler angles (roll, pitch, yaw)
        r = R.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])  # (QX, QY, QZ, QW)
        euler_angles = r.as_euler('xyz', degrees=False)
        current_yaw = (euler_angles[2])  # Yaw is the third Euler angle (rotation around Z-axis)
        print(f"Current yaw: {(current_yaw)} radians | Target yaw: {(self._target_yaw)} radians")

        # If no command is active, publish camera data and then pause the simulation
        if self._current_command is None:
            print("No command found, publishing camera data and pausing simulation.")
            self.publish_camera_data()  # Publish camera data right before pausing
            self._world.pause()
            return

        FORWARD_DISTANCE = 0.5  # scene units
        TURN_ANGLE = 0.52  # radians

        # If a new command is given, calculate the target position/yaw
        if self._current_command == "move_forward" and self._target_position is None:
            direction = np.array([np.cos(current_yaw), np.sin(current_yaw)])  # Direction robot is facing
            self._target_position = position[:2] + direction * FORWARD_DISTANCE  # Move forward by 1 unit
            print(f"Target position set: {self._target_position}")

        elif self._current_command == "turn_left" and self._target_yaw is None:
            self._target_yaw = self.normalize_angle(current_yaw + TURN_ANGLE)  # Turn left 30 degrees
            print(f"Initial yaw: {current_yaw} radians, Target yaw set: {self._target_yaw} radians")

        elif self._current_command == "turn_right" and self._target_yaw is None:
            self._target_yaw = self.normalize_angle(current_yaw - TURN_ANGLE)  # Turn right 30 degrees
            print(f"Initial yaw: {current_yaw} radians, Target yaw set: {self._target_yaw} radians")

        # Move the robot forward until the target position is reached
        if self._current_command == "move_forward" and self._target_position is not None:
            current_position = position[:2]  # Current 2D position of the robot
            distance = np.linalg.norm(self._target_position - current_position)

            if distance <= 0.1:  # Target reached within tolerance
                print("Reached target position.")
                self._current_command = None
                self._target_position = None
            else:
                throttle, steering = 0.5, 0  # Move forward
                print(f"Moving forward: distance to target = {distance}")
                self._jetbot.apply_wheel_actions(self._jetbot_controller.forward([throttle, steering]))

        # Rotate the robot until the target yaw is reached
        elif self._current_command == "turn_left" and self._target_yaw is not None:
            yaw_diff = np.abs(current_yaw - self._target_yaw)

            if yaw_diff <= np.radians(1):  # Target yaw reached within 1 degree
                print("Reached target yaw (left).")
                self._current_command = None
                self._target_yaw = None
            else:
                throttle, steering = 0, 0.5  # Turn left
                print(f"Turning left: yaw difference = {yaw_diff} radians")
                self._jetbot.apply_wheel_actions(self._jetbot_controller.forward([throttle, steering]))

        elif self._current_command == "turn_right" and self._target_yaw is not None:
            yaw_diff = np.abs(current_yaw - self._target_yaw)

            if yaw_diff <= np.radians(1):  # Target yaw reached within 1 degree
                print("Reached target yaw (right).")
                self._current_command = None
                self._target_yaw = None
            else:
                throttle, steering = 0, -1  # Turn right
                print(f"Turning right: yaw difference = {yaw_diff} radians")
                self._jetbot.apply_wheel_actions(self._jetbot_controller.forward([throttle, steering]))



    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._start_time = time.time()
        self._current_command = None
        self._target_position = None
        self._target_yaw = None
        self._prev_position = None
        self._prev_yaw = None

    def world_cleanup(self):
        # Ensure ROS2 is shut down when the world is cleaned up
        self.executor.shutdown()
        self.ros2_thread.join()
        if rclpy.ok():
            rclpy.shutdown()
