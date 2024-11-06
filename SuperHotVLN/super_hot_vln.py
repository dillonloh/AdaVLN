# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import json
import random
import time

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

enable_extension("omni.anim.people") 

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

class SuperHotVLN(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._moving_objects = []
        self._input_usd_path = None
        self._task_details_path = None
        self._task_details_list = None
        self._task_num = 0
        self._current_task = None

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
        
        matterport_env_usd = self._input_usd_path
        task_details_path = self._task_details_path
        with open(task_details_path, "r") as f:
            self._task_details_list = json.load(f)
        
        self._current_task = self._task_details_list[self._task_num]

        print(f"Task details: {self._task_details_list}")
        print(f"Current task: {self._current_task}")
        print(f"Loading Matterport environment from: {matterport_env_usd}")
        
        matterport_env_prim_path = "/World"

        add_reference_to_stage(usd_path=matterport_env_usd, prim_path=matterport_env_prim_path)
        assets_root_path = get_assets_root_path()

        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
        action.execute()
        
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot_prim_path = "/World/Jetbot"
        start_position = self._current_task["start_position"]
        start_orientation = rotvecs_to_quats([0, 0, self._current_task["heading"]])
        print(f"Start position: {start_position} | Start orientation: {start_orientation}")

        # Add the Jetbot robot
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

        # Add the Jetbot camera
        camera_prim_path = "/World/Jetbot/chassis/rgb_camera/jetbot_camera"

        world.scene.add(
            Camera(
                prim_path=camera_prim_path,
                name="jetbot_camera",
                resolution=(1280, 720),
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

        cmd_lines = ["Spawn Tom -9 0 0 0", "Tom GoToLoop -3 0 0 _ -6 0 0 _"] 
        
        load_characters(cmd_lines)
        print("Characters loaded.")
        print("Load commands into UI")
        CommandTextWidget.textbox_commands = "\n".join(cmd_lines)
        print(CommandTextWidget.textbox_commands)
        print("Commands loaded into UI.")
        print()
        print("Setting up characters...")  
        setup_characters()
        print("Characters set up.")
        

    def publish_camera_data(self):
        print("Publishing camera data...")
        # Capture RGB and Depth images from the camera
        rgb_data = self._camera.get_rgb()
        depth_data = self._camera.get_depth()

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

        # Handle the stop command: stop the robot and pause simulation
        if self._current_command == "stop":
            print("Stop command received. Stopping robot and pausing simulation.")
            self._jetbot.apply_wheel_actions(self._jetbot_controller.forward([0.0, 0.0]))  # Stop the robot's wheels
            self.publish_camera_data()  # Publish final camera data
            self._world.pause()  # Pause the simulation
            self._current_command = None  # Reset command
            return
        
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


## anim use


PERSISTENT_SETTINGS_PREFIX = "/persistent"
class PeopleSettings:
    COMMAND_FILE_PATH = "/exts/omni.anim.people/command_settings/command_file_path"
    ROBOT_COMMAND_FILE_PATH = "/exts/omni.anim.people/command_settings/robot_command_file_path"
    DYNAMIC_AVOIDANCE_ENABLED = "/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled"
    NAVMESH_ENABLED = "/exts/omni.anim.people/navigation_settings/navmesh_enabled"
    CHARACTER_ASSETS_PATH = f"{PERSISTENT_SETTINGS_PREFIX}/exts/omni.anim.people/asset_settings/character_assets_path"
    BEHAVIOR_SCRIPT_PATH = f"{PERSISTENT_SETTINGS_PREFIX}/exts/omni.anim.people/behavior_script_settings/behavior_script_path"
    CHARACTER_PRIM_PATH = f"{PERSISTENT_SETTINGS_PREFIX}/exts/omni.anim.people/character_prim_path"
    

def load_characters(cmd_lines, character_root_path="/World/Characters", assets_root_path=None):
    """
    Loads characters into the USD stage based on the commands provided in a specified file or a command textbox.

    Args:
        cmd_lines (list): List of command lines to interpret and initialize characters.
        character_root_path (str): The USD stage path where characters will be loaded.
        assets_root_path (str): Root path to the character assets; if None, attempts to fetch from Isaac Sim assets.
    """
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    
    print(f"Command lines: {cmd_lines}")
    # Initialize characters based on the extracted commands
    init_characters(stage, cmd_lines)

def init_characters(stage, cmd_lines):
    """
    Initializes characters on the USD stage based on command lines provided.

    Args:
        stage: The USD stage object where characters will be initialized.
        cmd_lines (list): List of command lines to interpret and initialize characters.
    """
    # Reset state from past simulation
    available_character_list = []
    spawned_agents_list = []
    setting_dict = carb.settings.get_settings()
    print("NAVMESH")
    print(setting_dict.get(PeopleSettings.NAVMESH_ENABLED))
    # Get root assets path from setting, if not set, get the Isaac Sim asset path
    people_asset_folder = setting_dict.get(PeopleSettings.CHARACTER_ASSETS_PATH)
    character_root_prim_path = setting_dict.get(PeopleSettings.CHARACTER_PRIM_PATH)
    if not character_root_prim_path:
        character_root_prim_path = "/World/Characters"

    if people_asset_folder:
        assets_root_path = people_asset_folder
    else:   
        root_path = get_assets_root_path()
        if root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        assets_root_path = "{}/Isaac/People/Characters".format(root_path)

    if not assets_root_path:
        carb.log_error("Could not find people assets folder")
    
    result, properties = omni.client.stat(assets_root_path)
    if result != omni.client.Result.OK:
        carb.log_error("Could not find people asset folder: " + str(assets_root_path))
        return

    if not Sdf.Path.IsValidPathString(character_root_prim_path):
        carb.log_error(str(character_root_prim_path) + " is not a valid character root prim's path")
    
    if not stage.GetPrimAtPath(character_root_prim_path):
        prims.create_prim(character_root_prim_path, "Xform")
    
    character_root_prim = stage.GetPrimAtPath(character_root_prim_path)
    # Delete all previously loaded agents
    for character_prim in character_root_prim.GetChildren():
        if character_prim and character_prim.IsValid() and character_prim.IsActive():
            prims.delete_prim(character_prim.GetPath())

    # Reload biped and animations
    default_biped_usd = "Biped_Setup"
    if not stage.GetPrimAtPath("{}/{}".format(character_root_prim_path, default_biped_usd)):
        biped_demo_usd = "{}/{}.usd".format(assets_root_path, default_biped_usd)
        prim = prims.create_prim("{}/{}".format(character_root_prim_path, default_biped_usd), "Xform", usd_path=biped_demo_usd)
        prim.GetAttribute("visibility").Set("invisible")

    # Reload character assets
    for cmd_line in cmd_lines:
        if not cmd_line:
            continue
        words = cmd_line.strip().split(' ')
        if words[0] != "Spawn":
            continue

        if len(words) != 6 and len(words) != 2:
            carb.log_error("Invalid 'Spawn' command issued, use command format - Spawn char_name or Spawn char_name x y z char_rotation.")
            return 

        # Add Spawn defaults
        if len(words) == 2:
            words.extend([0] * 4)

        # Do not use biped demo as a character name
        if str(words[1]) == "biped_demo":
            carb.log_warn("biped_demo is a reserved name, it cannot be used as a character name.")
            continue

        # Don't allow duplicates
        if str(words[1]) in spawned_agents_list:
            carb.log_warn(str(words[1]) + " has already been generated")
            continue

        # Check if prim already exists
        character_path = "{}/{}".format(character_root_prim_path, words[1])
        if stage.GetPrimAtPath(character_path):
            carb.log_warn("Path: " + character_path + " has been taken, please try another character name")
            continue

        char_name, char_usd_file = get_path_for_character_prim(assets_root_path, words[1], available_character_list)
        if char_usd_file:
            prim = prims.create_prim(character_path, "Xform", usd_path=char_usd_file)
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(words[2]), float(words[3]), float(words[4])))
            orient_attr = prim.GetAttribute("xformOp:orient")
            if isinstance(orient_attr.Get(), Gf.Quatf):
                orient_attr.Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(words[5])).GetQuat()))
            else:
                orient_attr.Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(words[5])).GetQuat())
            
            spawned_agents_list.append(words[1])

def get_path_for_character_prim(assets_root_path, agent_name, available_character_list):
    """
    Retrieves the USD path for a character's asset from the asset folder.

    Args:
        assets_root_path (str): Path to the root folder of character assets.
        agent_name (str): Name of the character to find.
        available_character_list (list): Cache of available character names.

    Returns:
        tuple: Character name (folder name) and the usd path to the character.
    """
    if not available_character_list:
        available_character_list = get_character_asset_list(assets_root_path)
        if not available_character_list:
            return None, None

    # Check if a folder with agent_name exists; if not, load a random character
    agent_folder = "{}/{}".format(assets_root_path, agent_name)
    result, properties = omni.client.stat(agent_folder)
    char_name = agent_name if result == omni.client.Result.OK else random.choice(available_character_list)
    
    character_folder = "{}/{}".format(assets_root_path, char_name)
    character_usd = get_usd_in_folder(character_folder)
    if not character_usd:
        return None, None
    
    if char_name in available_character_list:
        available_character_list.remove(char_name)
    
    return char_name, "{}/{}".format(character_folder, character_usd)

def get_character_asset_list(assets_root_path):
    """
    Retrieves a list of character directories in the asset folder.

    Args:
        assets_root_path (str): Path to the root folder of character assets.

    Returns:
        list: List of character names (folder names) found in the asset folder.
    """
    result, folder_list = omni.client.list("{}/".format(assets_root_path))
    if result != omni.client.Result.OK:
        carb.log_error("Unable to get character assets from provided asset root path.")
        return []

    return [
        folder.relative_path for folder in folder_list
        if (folder.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN) and not folder.relative_path.startswith(".")
    ]

def get_usd_in_folder(character_folder_path):
    """
    Finds the first USD file in a specified folder.

    Args:
        character_folder_path (str): Path to the folder to search.

    Returns:
        str or None: The name of the first USD file found, or None if none found.
    """
    result, folder_list = omni.client.list(character_folder_path)
    if result != omni.client.Result.OK:
        carb.log_error(f"Unable to read character folder path at {character_folder_path}")
        return None

    for item in folder_list:
        if item.relative_path.endswith(".usd"):
            return item.relative_path

    carb.log_error(f"No USD file found in {character_folder_path} character folder.")
    return None

def setup_characters():
    stage = omni.usd.get_context().get_stage()
    anim_graph_prim = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "AnimationGraph":
            anim_graph_prim = prim
            break

    if anim_graph_prim is None:
        carb.log_warn("Unable to find an animation graph on stage.")
        return

    for prim in stage.Traverse():
        if prim.GetTypeName() == "SkelRoot" and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
            omni.kit.commands.execute(
                "RemoveAnimationGraphAPICommand",
                paths=[Sdf.Path(prim.GetPrimPath())]
            )

            omni.kit.commands.execute(
                "ApplyAnimationGraphAPICommand",
                paths=[Sdf.Path(prim.GetPrimPath())],
                animation_graph_path=Sdf.Path(anim_graph_prim.GetPrimPath())
            )
            omni.kit.commands.execute(
                "ApplyScriptingAPICommand",
                paths=[Sdf.Path(prim.GetPrimPath())]
            )
            attr = prim.GetAttribute("omni:scripting:scripts")

            setting_dict = carb.settings.get_settings()
            ext_path = setting_dict.get(PeopleSettings.BEHAVIOR_SCRIPT_PATH)
            if not ext_path:
                ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__) + "/omni/anim/people/scripts/character_behavior.py"
                # temporary workaround because idk the api to get root path of isaac sim
                ext_path = ext_path.replace("exts/omni.isaac.examples", "extscache/omni.anim.people-0.5.0")
            print(f"Setting up character behavior script: {ext_path}")
            attr.Set([r"{}".format(ext_path)])
