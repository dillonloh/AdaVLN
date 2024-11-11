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
from omni.isaac.core.utils.rotations import euler_angles_to_quat
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
        self._task_details_path = "/home/dillon/0Research/VLNAgent/example_dataset/tasks/GLAQ4DNUx5U.json"
        self._task_details_list = None
        self._episode_number = 1
        self._current_task = None


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
        
        start_position = self._current_task["start_position"]
        start_orientation = self._current_task["start_rotation"]

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
                resolution=(1280, 720)
            )
        )

        self.setup_replicator_writers()

        # setup database

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


        self._jetbot_controller = DifferentialController(name="jetbot_control", wheel_radius=0.035, wheel_base=0.1)
        self._current_command = None
        self._target_position = None
        self._target_yaw = None
        
        cmd_lines = generate_cmd_lines(self._current_task['humans'])
        
        load_characters(cmd_lines)
        CommandTextWidget.textbox_commands = "\n".join(cmd_lines)
        setup_characters()

        self.ros2_node.publish_current_task_instruction(self._current_task["instruction"]["instruction_text"])

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
        position, orientation_quat = self._jetbot.get_world_pose()
        r = R.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])
        current_yaw = r.as_euler('xyz', degrees=False)[2]

        if self._current_command == "stop":
            handle_stop_command(self._jetbot, self._jetbot_controller, self._world)
            self._current_command = None
            self._world.pause() # we are done with this episode
            self._generate_results()
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


    # Function to generate the statistics
    def generate_results(self):
        # Function to calculate the distance between two points
        def calculate_distance(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))

        # Fetch the rows for the given episode
        all_rows = WorldState.select().where(WorldState.episode_id == self._episode_number)

        # Initialize variables for statistics
        total_distance = 0.0
        last_position = None
        robot_final_pos = None
        goal_pos = self._current_task["goals"][0]["position"]
        goal_radius = self.current_task["goals"][0]["radius"]
        collision = False
        oracle_success = False
        robot_positions = []

        for row in all_rows:
            robot_pos = np.array([row.robot_x, row.robot_y])
            robot_positions.append(robot_pos)

            # Calculate total distance moved by the robot
            if last_position is not None:
                total_distance += calculate_distance(last_position, robot_pos)
            
            # Check if the robot is within 1 unit of any character
            if row.characters:
                for character in row.characters:
                    character_pos = np.array([character['pos_x'], character['pos_y']])
                    if calculate_distance(robot_pos, character_pos) <= 1.0:
                        collision = True
            
            # Check if the robot is within 3 meters of the goal
            if calculate_distance(robot_pos, goal_pos) <= goal_radius:
                oracle_success = True

            last_position = robot_pos  # Update the last position

        # Get final robot position
        if robot_positions:
            robot_final_pos = robot_positions[-1]

        # 1) Distance from goal: Compare robot final pos with goal pos
        final_distance_from_goal = calculate_distance(robot_final_pos, goal_pos)

        # 2) Total distance moved by robot is already calculated in `total_distance`

        # 3) Collision with character: Already checked during iteration

        # 4) Final success: Robot is within 3 meters of the goal and no collisions
        final_success = (final_distance_from_goal <= goal_radius and not collision)

        # 5) Oracle success rate: Whether the robot was within 3m from goal at ANY point
        oracle_success_rate = oracle_success

        # Print the results
        print("####### Episode Statistics #######\n")
        print(f"Episode {self._episode_number} Statistics:")
        print(f"1) Final Distance from Goal: {final_distance_from_goal:.2f} meters")
        print(f"2) Total Distance Moved by Robot: {total_distance:.2f} meters")
        print(f"3) Collision with Character: {'Yes' if collision else 'No'}")
        print(f"4) Final Success (Within 3m from goal and no collisions): {'Yes' if final_success else 'No'}")
        print(f"5) Oracle Success Rate (Ever within 3m from goal): {'Yes' if oracle_success_rate else 'No'}")
        print("####### End of Episode Statistics #######\n\n")