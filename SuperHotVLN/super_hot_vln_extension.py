# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import os
import omni.ui as ui
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.ui.ui_utils import btn_builder, str_builder
from omni.isaac.examples.SuperHotVLN.super_hot_vln import SuperHotVLN

class SuperHotVLNExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="SuperHotVLNExtension",
            title="RobotCommand Controller",
            doc_link="",
            overview="SuperHotVLNExtension allows you to control the robot via commands like move_forward, turn_left, and turn_right.",
            sample=SuperHotVLN(),
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=1,
        )
        self.task_ui_elements = {}
        frame = self.get_frame(index=0)
        self.build_task_controls_ui(frame)
        return

    def _on_move_forward_button_event(self):
        # Set the command to move forward and start the simulation
        self.sample._current_command = "move_forward"
        self.sample._world.play()  # Ensure simulation starts or resumes
        return

    def _on_turn_left_button_event(self):
        # Set the command to turn left and start the simulation
        self.sample._current_command = "turn_left"
        self.sample._world.play()  # Ensure simulation starts or resumes
        return

    def _on_turn_right_button_event(self):
        # Set the command to turn right and start the simulation
        self.sample._current_command = "turn_right"
        self.sample._world.play()  # Ensure simulation starts or resumes
        return

    def _input_usd_path_event(self, val):
        self.sample._input_usd_path = val.get_value_as_string()
        return
    
    def post_reset_button_event(self):
        self.task_ui_elements["Move Forward"].enabled = True
        self.task_ui_elements["Turn Left"].enabled = True
        self.task_ui_elements["Turn Right"].enabled = True
        return

    def build_task_controls_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                # Update the Frame Title
                frame.title = "Command Controls"
                frame.visible = True

                # Move Forward Button
                move_forward_dict = {
                    "label": "Move Forward",
                    "type": "button",
                    "text": "Move Forward",
                    "tooltip": "Move the robot forward",
                    "on_clicked_fn": self._on_move_forward_button_event,
                }
                self.task_ui_elements["Move Forward"] = btn_builder(**move_forward_dict)
                self.task_ui_elements["Move Forward"].enabled = True

                # Turn Left Button
                turn_left_dict = {
                    "label": "Turn Left",
                    "type": "button",
                    "text": "Turn Left",
                    "tooltip": "Turn the robot 30 degrees left",
                    "on_clicked_fn": self._on_turn_left_button_event,
                }
                self.task_ui_elements["Turn Left"] = btn_builder(**turn_left_dict)
                self.task_ui_elements["Turn Left"].enabled = True

                # Turn Right Button
                turn_right_dict = {
                    "label": "Turn Right",
                    "type": "button",
                    "text": "Turn Right",
                    "tooltip": "Turn the robot 30 degrees right",
                    "on_clicked_fn": self._on_turn_right_button_event,
                }
                self.task_ui_elements["Turn Right"] = btn_builder(**turn_right_dict)
                self.task_ui_elements["Turn Right"].enabled = True
                
                dict = {
                    "label": "Input USD Path",
                    "type": "stringfield",
                    "tooltip": "Input USD Path",
                    "on_clicked_fn": self._input_usd_path_event,
                    "use_folder_picker": True,
                    "read_only": False,
                }
                self.task_ui_elements["Input USD Path"] = str_builder(**dict)