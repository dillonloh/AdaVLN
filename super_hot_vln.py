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

import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController, WheelBasePoseController
from omni.isaac.core.objects import VisualCuboid
import carb


class SuperHotVLN(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._moving_objects = []
        return

    def setup_scene(self):

        world = self.get_world()

        matterport_env_usd = "/home/dillon/0Research/IsaacSim/test.usd"
        matterport_env_prim_path = "/World"

        add_reference_to_stage(usd_path=matterport_env_usd, prim_path=matterport_env_prim_path)
        assets_root_path = get_assets_root_path()

        # jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot_prim_path = "/World/Jetbot"
        
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
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        self._world.add_physics_callback("moving_objects", callback_fn=self.move_objects_in_random_paths)
        self._jetbot_controller = DifferentialController(name="jetbot_control", wheel_radius=0.035, wheel_base=0.1)
        self._current_command = None
        self._target_position = None
        self._target_yaw = None

    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def move_objects_in_random_paths(self, step_size):
        # Move each object in a random direction with a small step size
        for i, obj in enumerate(self._moving_objects):
            # Generate a random direction and move the object slightly
            random_direction = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0])
            current_position, current_orientation = obj.get_world_pose()
            new_position = np.array(current_position) + random_direction
            print(f"Object {i+1} current position: {current_position}")
            obj.set_world_pose(new_position.tolist(), current_orientation)

    def send_robot_actions(self, step_size):
        # Get the current position and orientation (in quaternion) of the robot
        position, orientation_quat = self._jetbot.get_world_pose()

        # Convert quaternion (QW, QX, QY, QZ) to Euler angles (roll, pitch, yaw)
        r = R.from_quat([orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]])  # (QX, QY, QZ, QW)
        euler_angles = r.as_euler('xyz', degrees=False)
        current_yaw = (euler_angles[2])  # Yaw is the third Euler angle (rotation around Z-axis)
        print(f"Current yaw: {(current_yaw)} radians | Target yaw: {(self._target_yaw)} radians")

        # If no command is active, pause the simulation
        if self._current_command is None:
            print("No command found, pausing simulation.")
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
        return
