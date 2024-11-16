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
import omni
import omni.ui as ui
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.ui.ui_utils import btn_builder, str_builder, int_builder
from omni.isaac.ui import IntField
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
    
    def _input_task_details_path_event(self, val):
        self.sample._task_details_path = val.get_value_as_string()
        return
    
    def _input_episode_number_event(self, val):
        self.sample._episode_number = int(val)
        return
    
    def _on_next_episode(self):
        async def _on_next_episode_async():
            await self._sample.load_next_episode()
            await omni.kit.app.get_app().next_update_async()
            self._sample._world.add_stage_callback("stage_event_1", self.on_stage_event)
            self._enable_all_buttons(True)
            self._buttons["Load World"].enabled = True
            self.post_load_button_event()
            self._sample._world.add_timeline_callback("stop_reset_event", self._reset_on_stop_event)

        asyncio.ensure_future(_on_next_episode_async())
        return
    
    def _on_load_episode(self):
        async def _on_load_episode_async():
            await self._sample.load_episode()
            await omni.kit.app.get_app().next_update_async()
            self._sample._world.add_stage_callback("stage_event_1", self.on_stage_event)
            self._enable_all_buttons(True)
            self._buttons["Load World"].enabled = True
            self.post_load_button_event()
            self._sample._world.add_timeline_callback("stop_reset_event", self._reset_on_stop_event)

        asyncio.ensure_future(_on_load_episode_async())
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
        
                # Load Next Episode Button
                stop_dict = {
                    "label": "Stop",
                    "type": "button",
                    "text": "Stop",
                    "tooltip": "Stop",
                    "on_clicked_fn": self._on_next_episode,
                }
                self.task_ui_elements["Stop"] = btn_builder(**stop_dict)
                self.task_ui_elements["Stop"].enabled = True
        
        
                dict = {
                    "label": "Input USD Path",
                    "type": "stringfield",
                    "tooltip": "Input USD Path",
                    "on_clicked_fn": self._input_usd_path_event,
                    "use_folder_picker": True,
                    "read_only": False
                }
                self.task_ui_elements["Input USD Path"] = str_builder(**dict)

                dict = {
                    "label": "Input Task Details Path",
                    "type": "stringfield",
                    "tooltip": "Input Task Details Path",
                    "on_clicked_fn": self._input_task_details_path_event,
                    "use_folder_picker": True,
                    "read_only": False
                }
                self.task_ui_elements["Input Task Details Path"] = str_builder(**dict)

                dict = {
                    "label": "Episode Number",
                    "tooltip": "Episode Number",
                    "default_value": 1,
                    "on_value_changed_fn": self._input_episode_number_event,
                }
                self.task_ui_elements["Episode Number"] = IntField(**dict)
                
                # Load Episode Button
                load_episode_dict = {
                    "label": "Load Episode",
                    "type": "button",
                    "text": "Load Episode",
                    "tooltip": "Load Episode",
                    "on_clicked_fn": self._on_load_episode,
                }
                self.task_ui_elements["Load Episode"] = btn_builder(**load_episode_dict)
                self.task_ui_elements["Load Episode"].enabled = True
                
                # Load Next Episode Button
                next_episode_dict = {
                    "label": "Next Episode",
                    "type": "button",
                    "text": "Next Episode",
                    "tooltip": "Next Episode",
                    "on_clicked_fn": self._on_next_episode,
                }
                self.task_ui_elements["Next Episode"] = btn_builder(**next_episode_dict)
                self.task_ui_elements["Next Episode"].enabled = True

