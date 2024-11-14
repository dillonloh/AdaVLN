import json
import random
import time
import math

import cv2  
import numpy as np
from scipy.spatial.transform import Rotation as R

import omni.anim.graph.core as ag
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.numpy import rotvecs_to_quats
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController, WheelBasePoseController
from omni.isaac.core.objects import VisualCuboid
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.kit.actions.core
from omni.physx import get_physx_scene_query_interface
from omni.physx.scripts import utils

import carb

from omni.anim.people.ui_components.command_setting_panel.command_text_widget import CommandTextWidget
from pxr import Sdf, Gf, UsdGeom
from omni.isaac.core.utils import prims

import threading

from .utils.dynamic_anim import *
from .utils.robot_movement import *
from .utils.transforms import *
from .database.db_utils import create_db
from .database.models.world_state import WorldState

enable_extension("omni.anim.people") 

import os
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, Writer
from PIL import Image


# Replicator Writer for saving frame data
class MyWriter(Writer):
    def __init__(self, rgb: bool = True, distance_to_camera: bool = False):
        self._frame_id = 0
        self.file_path = os.path.join(os.getcwd(), "_out_mc_writer", "")
        os.makedirs(self.file_path, exist_ok=True)

        # Register annotators based on flags
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        if distance_to_camera:
            self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_camera"))

    def write(self, data):
        pass
        self._frame_id += 1
        
class SuperHotVLN(BaseSample):
    def __init__(self) -> None:

        super().__init__()

        settings = carb.settings.get_settings()
        settings.set("exts/omni.anim.people/navigation_settings/navmesh_enabled", False)
        settings.set("exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
        self._input_usd_dir = "/home/dillon/0Research/VLNAgent/example_dataset/merged"
        self._input_usd_path = "/home/dillon/0Research/VLNAgent/example_dataset/merged/GLAQ4DNUx5U.usd"
        self._task_details_path = "/home/dillon/0Research/VLNAgent/example_dataset/tasks/tasks.json"
        self._task_details_list = None
        self._episode_number = 1
        self._current_task = None
        self._db = None

    def setup_scene(self):
        
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from cv_bridge import CvBridge  
        from .utils.ros2_publisher import ROS2PublisherNode
    
        if not rclpy.ok():
            rclpy.init()
            
        self.ros2_node = ROS2PublisherNode(self)

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.ros2_node)
        self.ros2_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros2_thread.start()

        world = self.get_world()
        
        with open(self._task_details_path, "r") as f:
            self._task_details_list = json.load(f).get("episodes")

        self._current_task = self._task_details_list[self._episode_number - 1]
        self._input_usd_path = f"{self._input_usd_dir}/{self._current_task['scene_id']}.usd"
        matterport_env_usd = self._input_usd_path

        matterport_env_prim_path = "/World"
        add_reference_to_stage(usd_path=matterport_env_usd, prim_path=matterport_env_prim_path)
        
        # add lighting (based on Default lighting)
        # Add Dome Light at position (0, 0, 305) with rotation (0, 0, 62.3)
        dome_light_prim_path = "/World/DomeLight"
        dome_light = prim_utils.create_prim(
            dome_light_prim_path,
            "DomeLight",
            position=np.array([0.0, 0.0, 305.0]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, 62.3]), degrees=True),  # Rotation in degrees
            attributes={
                "inputs:exposure": 9.0,  # Adjust intensity as needed
                "inputs:intensity": 1.0,  # Adjust intensity as needed
                "inputs:color": (1.0, 1.0, 1.0),  # White light color
            }
        )
        dome_light.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)

        # Add Distant Light at position (0, 0, 305) with rotation (55, 0, 135)
        distant_light_prim_path = "/World/DistantLight"
        distant_light = prim_utils.create_prim(
            distant_light_prim_path,
            "DistantLight",
            position=np.array([0.0, 0.0, 305.0]),
            orientation=euler_angles_to_quat(np.array([55.0, 0.0, 135.0]), degrees=True),  # Rotation in degrees
            attributes={
                "inputs:exposure": 10.0,  # Adjust intensity as needed
                "inputs:intensity": 1.0,  # Adjust intensity as needed
                "inputs:color": (1.0, 1.0, 1.0),  # White light color
                "inputs:angle": 0.53,  # White light color
            }
        )

        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot_prim_path = "/World/Jetbot"
        
        start_position = transform_to_sim_position(self._current_task["start_position"])
        start_orientation = transform_to_sim_rotation(self._current_task["start_rotation"])
        print(f"Start position: {start_position}, Start orientation: {start_orientation}")

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
        camera = Camera(
                prim_path=camera_prim_path,
                name="jetbot_camera",
                resolution=(1280, 720),
            )
        world.scene.add(camera)
        # set clipping range lower bound small to prevent clipping when robot collide with wall
        camera.set_clipping_range(0.00001, 1000000)
        camera_xform_path = "/World/Jetbot/chassis/rgb_camera"
        camera_xform = world.stage.GetPrimAtPath(camera_xform_path)

        # This is a hardcoded local transform that works well for pointing the camera straight forward (slight tilt up)
        new_matrix = Gf.Matrix4d(
            (0.9739781364270557, 5.967535426946605e-7, 0.226652306401774, 0),
            (-6.934631969087326e-7, 1.0000015028778466, 8.558670747042332e-7, 0),
            (-0.22665111823320652, -0.0000013076089426975306, 0.9739772275779269, 0),
            (0.046500139001482164, -3.386000451406582e-8, 0.06720042363232537, 1)
        )

        # Set the new matrix as the camera transform
        camera_xform.GetAttribute("xformOp:transform").Set(new_matrix)

        self.setup_replicator_writers()

        # setup database

        if not self._db:
            self._db = create_db()

        self._db.connect(reuse_if_open=True)
        self._db.create_tables([WorldState])

    def setup_replicator_writers(self):
        # Register custom writer and randomizer
        rep.WriterRegistry.register(MyWriter)

        self._camera_rp = []
        
        # Define render products
        rp = rep.create.render_product("/World/Jetbot/chassis/rgb_camera/jetbot_camera", resolution=(1280, 720))
        self._camera_rp.extend([rp])
        
        self._writer = rep.WriterRegistry.get("MyWriter")
        self._writer.initialize(rgb=True, distance_to_camera=True)
        self._writer.attach([rp])

        self._rgb = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self._distance_to_camera = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self._rgb.attach(rp)
        self._distance_to_camera.attach(rp)

    async def setup_post_load(self):

        self._start_time = time.time()
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("jetbot")
    

        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        self._world.add_physics_callback("checking_collisions", callback_fn=self.check_collision)
        self._world.add_physics_callback("storing_data", callback_fn=self.store_data)

        self._current_command = None
        self._target_position = None
        self._target_yaw = None
        self._collision = None
        
        # Initialize attributes for timeout and collision tracking
        self.timeout_duration = None
        self.elapsed_time = 0.0  # Track elapsed time for timeouts
        self._collision_with_building = False

        # Configurable speeds and distances
        self.linear_speed = 0.5
        self.rotation_speed = np.radians(30)
        self.move_distance = 0.25
        self.rotation_angle = np.radians(15)

        # Initialize the Differential Controller with speed limits
        self.wheel_radius = 0.035  # wheel radius in meters
        self.wheel_base = 0.1  # distance between wheels in meters

        self._jetbot_controller = DifferentialController(name="jetbot_control", wheel_radius=self.wheel_radius, wheel_base=self.wheel_base)
        self._current_command = None
        self._initial_position = None
        self._initial_yaw = None
        
        self._collision = None

        cmd_lines = generate_cmd_lines(self._current_task['humans'])
        
        load_characters(cmd_lines)
        CommandTextWidget.textbox_commands = "\n".join(cmd_lines)
        setup_characters()

        self.ros2_node.publish_current_task_instruction(self._current_task["instruction"]["instruction_text"])
        self.ros2_node.publish_episode_number(self._episode_number)
        
        characters_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Characters")
        building_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Building")
        print(building_prim)
        self.add_boundingcube_collision_to_meshes(characters_prim)
        self.add_boundingcube_collision_to_meshes(building_prim)
        self.disable_shadow_casting(building_prim)

        await self._world.reset_async()

        print(f"Loaded scene with details: {self._current_task}")

        return 
    
    def add_boundingcube_collision_to_meshes(self, prim, approximationShape=None):
        """
        Recursively add bounding cube colliders to all UsdGeom.Mesh children under the given prim.
        """
        for child in prim.GetChildren():
            # Check if the child prim is a UsdGeom.Mesh and that it isnt /World/Characters/Biped_Setup
            if child.IsA(UsdGeom.Mesh) and not child.GetPath().pathString.startswith("/World/Characters/Biped_Setup"):
                # Set collider with bounding cube approximation
                if approximationShape is None:
                    utils.setCollider(child)
                    print(f"Collider added to {child.GetPath().pathString}")
                else:
                    utils.setCollider(child, approximationShape=approximationShape)
                    print(f"Collider added to {child.GetPath().pathString} with {approximationShape} approximation")
            
            # Recursively call this function to process deeper levels
            self.add_boundingcube_collision_to_meshes(child)

    def disable_shadow_casting(self, prim):
        """
        Recursively disable shadow casting to all UsdGeom.Mesh children under the given prim.
        """
        for child in prim.GetChildren():
            
            child.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool)
            child.GetAttribute("primvars:doNotCastShadows").Set(True)

    def publish_camera_data(self):
        # Using replicator's annotation for image data capture
        rgb_data = self._rgb.get_data()
        depth_data = self._distance_to_camera.get_data()
        
        self.ros2_node.publish_camera_data(rgb_data, depth_data)

    def send_robot_actions(self, step_size):

        if self._current_command == "stop":
            handle_stop_command(self._jetbot, self._jetbot_controller, self._world)
            self._current_command = None
            self._world.pause() # we are done with this episode
            # self.generate_results()
            return
        
        if self._current_command is None:
            self.publish_camera_data()
            self._world.pause()
            return

        # Enforce a 2-second timeout if colliding with the building
        if self._collision_with_building:
            print("Collision with building detected. Enforcing 2-second timeout.")
            if self.elapsed_time == 0.0:
                self.timeout_duration = 2.0
            self.elapsed_time += step_size
            print(f"Elapsed time: {self.elapsed_time:.2f}s")
            
            if self.elapsed_time >= self.timeout_duration:
                print("Timeout reached due to collision with building. Stopping action.")
                self.publish_camera_data()
                handle_stop_command(self._jetbot, self._jetbot_controller, self._world)
                self._current_command = None
                self._collision_with_building = False
                self.elapsed_time = 0.0
                return
        else:
            self.elapsed_time = 0.0

        # Get the current position and yaw of the robot
        position, orientation_quat = self._jetbot.get_world_pose()
        r = R.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])
        current_yaw = r.as_euler('xyz', degrees=False)[2]
        current_position = np.array(position[:2])

        # Set the initial position and yaw if starting a new command
        if self._initial_position is None and self._initial_yaw is None:
            self._initial_position = current_position
            self._initial_yaw = current_yaw

        # Process robot commands based on distance or rotation from initial values
        if self._current_command == "move_forward":
            if handle_robot_move_command(self._initial_position, current_position, self._jetbot, self._jetbot_controller, self.linear_speed, self.move_distance):
                self._current_command = None
                self._initial_position = None
                self._initial_yaw = None

        elif self._current_command == "turn_left":
            if handle_robot_turn_command(self._initial_yaw, current_yaw, "left", self._jetbot, self._jetbot_controller, self.rotation_speed, self.rotation_angle):
                self._current_command = None
                self._initial_position = None
                self._initial_yaw = None

        elif self._current_command == "turn_right":
            if handle_robot_turn_command(self._initial_yaw, current_yaw, "right", self._jetbot, self._jetbot_controller, self.rotation_speed, self.rotation_angle):
                self._current_command = None
                self._initial_position = None
                self._initial_yaw = None

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

        def report_raycast(hit):
            """
            Callback for handling raycast hits. Records collision details and stops further raycasting.
            """
            # Check for collision type (e.g., with a character or building)
            if hit.collision.startswith("/World/Characters"):
                self.collisions.append({"collision": hit.collision})
                character_name = hit.collision.split("/")[-1]  # Extract character name
                print(f"Collision detected with a human: {character_name}")
                self._collision = True
                return False
            elif hit.collision.startswith("/World/Building"):
                self.collisions.append({"collision": hit.collision})
                print("Collision detected with the building.")
                self._collision = True
                self._collision_with_building = True  # Flag collision with building
                return False
            return True  # continue finding other collisions
        
        # Perform raycasting in a circular pattern around the robot
        for i in range(num_rays):
            angle_rad = math.radians(i * angle_step)
            direction = carb.Float3(math.cos(angle_rad), math.sin(angle_rad), 0.0)

            # Perform raycast in the specified direction
            get_physx_scene_query_interface().raycast_all(origin, direction, ray_distance, report_raycast)

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

    def store_data(self, step_size):
        """
        Stores various world state data for analysis later. These include:
        - Robot position and orientation
        - Human positions
        - Actions taken by the robot
        - Action waypoints
        """

        # Retrieve robot's position and orientation
        position, orientation_quat = self._jetbot.get_world_pose()
        robot_x, robot_y, robot_z = position
        r = R.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])
        robot_yaw = r.as_euler('xyz', degrees=False)[2]

        # Simulation time
        sim_time = time.time() - self._start_time

        # Gather all character positions as a list of dictionaries
        character_positions = []
        characters = ag.get_characters()  # Assuming ag.get_characters() returns a list of character objects

        for index, character in enumerate(characters):
            pos = carb.Float3(0, 0, 0)
            rot = carb.Float4(0, 0, 0, 0)
            character.get_world_transform(pos, rot)  # Retrieve the position and rotation

            # If position data is available, store it; otherwise, skip or log missing data
            if pos:
                character_data = {
                    "character_id": f"character_{index}",  # Replace with actual character ID if available
                    "pos_x": pos.x,
                    "pos_y": pos.y,
                    "pos_z": pos.z
                }
                character_positions.append(character_data)
                print(f"Character {index} position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
            else:
                print(f"Character {index} position data not available.")

        # Create the WorldState entry with characters stored as JSON
        world_state = WorldState.create(
            scene_id=self._input_usd_path.split("/")[-1].split(".")[0],  # Extract environment ID from USD file name
            episode_id=str(self._episode_number),
            sim_time=sim_time,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_z=robot_z,
            robot_yaw=robot_yaw,
            characters=character_positions  # Store the list of characters as JSON
        )

        # Print a summary of the stored state
        print(f"Stored World State at time {sim_time:.2f}s - Robot: ({robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f}), Yaw: {robot_yaw:.2f}")
        print(f"Included {len(character_positions)} characters in the state.")

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._start_time = time.time()
        self._current_command = None
        self._target_position = None
        self._target_yaw = None
        self._prev_position = None
        self._prev_yaw = None
        self._collision = None

    def world_cleanup(self):
        import rclpy

        print("cleaning up world")
        self.executor.shutdown()
        self.ros2_thread.join()
        if rclpy.ok():
            rclpy.shutdown()
        
        self._db.close()

    # change to next episode
    async def load_next_episode(self):
        if self._episode_number >= len(self._task_details_list):
            print("Max episode reached")
            return
        self._episode_number += 1
        self._current_task = self._task_details_list[self._episode_number - 1]
        self._input_usd_path = f"{self._input_usd_dir}/{self._current_task['scene_id']}.usd"
        await self.load_world_async()

    # change to previous episode
    async def load_previous_episode(self):
        if self._episode_number <= 1:
            print("Min episode reached")
            return
        self._episode_number -= 1
        self._current_task = self._task_details_list[self._episode_number - 1]
        self._input_usd_path = f"{self._input_usd_dir}/{self._current_task['scene_id']}.usd"
        await self.load_world_async()

    # change to next episode
    async def load_episode(self):
        if self._task_details_list is None:
            with open(self._task_details_path, "r") as f:
                self._task_details_list = json.load(f).get("episodes")

        if self._episode_number >= len(self._task_details_list):
            print(f"Invalid episode number {self._episode_number}")
            raise ValueError(f"Invalid episode number {self._episode_number}")
        
        self._current_task = self._task_details_list[self._episode_number - 1]
        self._input_usd_path = f"{self._input_usd_dir}/{self._current_task['scene_id']}.usd"
        await self.load_world_async()
