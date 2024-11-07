import json
import random
import time
import math

import cv2  
import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.numpy import rotvecs_to_quats
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController, WheelBasePoseController
from omni.isaac.core.objects import VisualCuboid
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.kit.actions.core
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts import utils

import carb

from omni.anim.people.ui_components.command_setting_panel.command_text_widget import CommandTextWidget
from pxr import Sdf, Gf, UsdGeom
from omni.isaac.core.utils import prims

import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge  

from .utils.dynamic_anim import *
from .utils.robot_movement import *
from .utils.ros2_publisher import ROS2PublisherNode

enable_extension("omni.anim.people") 

class SuperHotVLN(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        settings = carb.settings.get_settings()
        settings.set("exts/omni.anim.people/navigation_settings/navmesh_enabled", False)
        settings.set("exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
        
        self._input_usd_path = "/home/dillon/0Research/VLNAgent/example_dataset/merged/GLAQ4DNUx5U.usd"
        self._task_details_path = "/home/dillon/0Research/VLNAgent/example_dataset/tasks/GLAQ4DNUx5U.json"
        self._task_details_list = None
        self._task_num = 0
        self._current_task = None

        if not rclpy.ok():
            rclpy.init()
        self.ros2_node = ROS2PublisherNode(self)

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.ros2_node)
        self.ros2_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros2_thread.start()

    def setup_scene(self):
        world = self.get_world()
        
        matterport_env_usd = self._input_usd_path
        task_details_path = self._task_details_path
        with open(task_details_path, "r") as f:
            self._task_details_list = json.load(f)
        
        self._current_task = self._task_details_list[self._task_num]
        matterport_env_prim_path = "/World"
        add_reference_to_stage(usd_path=matterport_env_usd, prim_path=matterport_env_prim_path)
        
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot_prim_path = "/World/Jetbot"
        
        start_position = self._current_task["start_position"]
        start_orientation = rotvecs_to_quats([0, 0, self._current_task["heading"]])

        world.scene.add(
            WheeledRobot(
                prim_path=jetbot_prim_path,
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                name="jetbot", 
                usd_path=jetbot_asset_path,
                position=start_position,
                orientation=start_orientation,
                create_robot=True
            )
        )

        camera_prim_path = "/World/Jetbot/chassis/rgb_camera/jetbot_camera"
        world.scene.add(
            Camera(
                prim_path=camera_prim_path,
                name="jetbot_camera",
                resolution=(1280, 720),
            )
        )
        
    async def setup_post_load(self):
        self._start_time = time.time()
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("jetbot")

        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        self._world.add_physics_callback("checking_collisions", callback_fn=self.check_collision)

        self._jetbot_controller = DifferentialController(name="jetbot_control", wheel_radius=0.035, wheel_base=0.1)
        self._current_command = None
        self._target_position = None
        self._target_yaw = None
        self._camera = self._world.scene.get_object("jetbot_camera")   
        
        self._camera.initialize()
        self._camera.add_motion_vectors_to_frame()
        self._camera.add_distance_to_image_plane_to_frame()
        self.bridge = CvBridge()

        cmd_lines = generate_cmd_lines(self._current_task['humans'])
        
        load_characters(cmd_lines)
        CommandTextWidget.textbox_commands = "\n".join(cmd_lines)
        setup_characters()

        characters_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Characters")
        self.add_boundingcube_collision_to_meshes(characters_prim)

        await self._world.reset_async()

    def add_boundingcube_collision_to_meshes(self, prim):
        """
        Recursively add bounding cube colliders to all UsdGeom.Mesh children under the given prim.
        """
        for child in prim.GetChildren():
            # Check if the child prim is a UsdGeom.Mesh and that it isnt /World/Characters/Biped_Setup
            if child.IsA(UsdGeom.Mesh) and not child.GetPath().pathString.startswith("/World/Characters/Biped_Setup"):
                # Set collider with bounding cube approximation
                utils.setCollider(child, approximationShape="boundingCube")
                print(f"Collider added to {child.GetPath().pathString}")
            
            # Recursively call this function to process deeper levels
            self.add_boundingcube_collision_to_meshes(child)

    def publish_camera_data(self):
        rgb_data = self._camera.get_rgb()
        depth_data = self._camera.get_depth()
        self.ros2_node.publish_camera_data(rgb_data, depth_data)

    def send_robot_actions(self, step_size):
        position, orientation_quat = self._jetbot.get_world_pose()
        r = R.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])
        current_yaw = r.as_euler('xyz', degrees=False)[2]

        if self._current_command == "stop":
            handle_stop_command(self._jetbot, self._jetbot_controller, self._world)
            self.publish_camera_data()
            self._current_command = None
            return
        
        if self._current_command is None:
            self.publish_camera_data()
            self._world.pause()
            return

        if self._current_command == "move_forward":
            self._target_position = handle_robot_move_command(position[:2], current_yaw, self._target_position, self._jetbot, self._jetbot_controller)
            if self._target_position is None:
                self._current_command = None

        elif self._current_command == "turn_left":
            self._target_yaw = handle_robot_turn_command(current_yaw, TURN_ANGLE=0.52, turn_direction="left", target_yaw=self._target_yaw, jetbot=self._jetbot, jetbot_controller=self._jetbot_controller)
            if self._target_yaw is None:
                self._current_command = None

        elif self._current_command == "turn_right":
            self._target_yaw = handle_robot_turn_command(current_yaw, TURN_ANGLE=0.52, turn_direction="right", target_yaw=self._target_yaw, jetbot=self._jetbot, jetbot_controller=self._jetbot_controller)
            if self._target_yaw is None:
                self._current_command = None

    def check_collision(self, step_size):
        """
        Checks for collisions using raycasting in a 360-degree pattern around the robot.
        Records details of any detected collisions.
        """
        # Clear any previous collision data
        self.collisions = []  # Using an instance variable to store collisions for this method

        # Get robot position and orientation
        position, orientation_quat = self._jetbot.get_world_pose()
        x, y, z = position[0], position[1], position[2]

        # Raycast parameters
        ray_distance = 0.5  # The maximum distance to check for obstacles
        num_rays = 36       # Number of rays in 360 degrees
        angle_step = 360 / num_rays
        origin = carb.Float3(x, y, z + 0.1)  # Start position slightly above ground

        collision_detected = False

        def report_raycast(hit):
            """
            Callback for handling raycast hits. Records collision details and stops further raycasting.
            """
            # Check for collision type (e.g., with a character or building)
            if hit.collision.startswith("/World/Characters"):
                # Append hit details to the collisions list
                self.collisions.append({"collision": hit.collision})
                character_name = hit.collision.split("/")[-1]  # Extract character name
                print(f"Collision detected with a human: {character_name}")
                return False
            elif hit.collision.startswith("/World/Building"):
                self.collisions.append({"collision": hit.collision})
                print("Collision detected with the building.")
                return False
            # If neither of the above, likely is internal collision so we ignore it
            return True # continue finding other collisions
        
        # Perform raycasting in a circular pattern around the robot
        for i in range(num_rays):
            if collision_detected:
                break  # Exit loop if a collision has been detected

            angle_rad = math.radians(i * angle_step)
            direction = carb.Float3(math.cos(angle_rad), math.sin(angle_rad), 0.0)

            # Perform raycast in the specified direction
            hit = get_physx_scene_query_interface().raycast_all(origin, direction, ray_distance, report_raycast)

            # Check if any collisions were recorded
            if self.collisions:
                collision_detected = True

        # Print collision details if any were detected
        if self.collisions:
            print("Collision details:")
            for i, collision in enumerate(self.collisions, start=1):
                print(f"Collision {i}:")
                for key, value in collision.items():
                    print(f"  {key}: {value}")
                print()  # Newline for readability
        else:
            print("No obstacles detected within the specified radius in the X-Y plane.")

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
        self.executor.shutdown()
        self.ros2_thread.join()
        if rclpy.ok():
            rclpy.shutdown()
