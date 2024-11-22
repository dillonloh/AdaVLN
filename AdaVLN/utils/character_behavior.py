# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations
from omni.kit.scripting import BehaviorScript
from omni.anim.people.scripts.utils import Utils
import carb
import omni.usd
import omni.anim.graph.core as ag
from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager
from omni.anim.people.scripts.global_queue_manager import GlobalQueueManager
from omni.anim.people.scripts.navigation_manager import NavigationManager
from omni.anim.people.scripts.commands.idle import * 
from omni.anim.people.scripts.commands.goto import *
from omni.anim.people.scripts.commands.look_around import *
from omni.anim.people.scripts.commands.queue import *
from omni.anim.people.scripts.commands.dequeue import *
from omni.anim.people.scripts.commands.sit import *
from omni.anim.people.scripts.commands.teleport import * #DILLON: added teleport command
from omni.anim.people.scripts.commands.waittilltime import * #DILLON: added waittilltime command
from omni.anim.people.scripts.commands.gotonew import * #DILLON: added new goto command with timeout
from omni.anim.people.scripts.commands.gotoloop import * #DILLON: added new goto command with loop

from omni.anim.people.ui_components import CommandTextWidget
from omni.anim.people import PeopleSettings
import importlib

print("my character_behaviour.py")

class CharacterBehavior(BehaviorScript):
    """
    Character controller class that reads commands from a command file and drives character actions.
    """    
    def on_init(self):
        """
        Called when a script is attached to characters and when a stage is loaded. Uses renew_character_state() to initialize character state.
        """
        self.renew_character_state()


    def on_play(self):
        """
        Called when entering runtime (when clicking play button). Uses renew_character_state() to initialize character state.
        """
        self.renew_character_state()
        self.current_time = 0.0

    def on_stop(self):
        """
        Called when exiting runtime (when clicking stop button). Uses on_destroy() to clear state.
        """
        self.on_destroy()

    def on_destroy(self):
        """
        Clears character state by deleting global variable instances.
        """
        self.character_name = None
        if self.character_manager is not None:
            self.character_manager.destroy()
            self.character_manager = None
        
        if self.navigation_manager is not None:
            self.navigation_manager.destroy()
            self.navigation_manager = None

        if self.queue_manager is not None:
            self.queue_manager.destroy()
            self.queue_manager = None

    def renew_character_state(self):
        """
        Defines character variables and loads settings.
        """
        self.setting =  carb.settings.get_settings()
        # print(f"Setting: {self.setting.get_settings_dictionary()}")
        self.command_path = self.setting.get(PeopleSettings.COMMAND_FILE_PATH)
        self.navmeshEnabled = False
        self.avoidanceOn = False
        self.character_name = self.get_character_name(str(self.prim_path))
        carb.log_info("Character name is {}".format(self.character_name))
        self.character = None
        self.current_command = None
        self.navigation_manager = None
        self.character_manager = None
        self.queue_manager = None
        
    def get_character_name(self, prim_path):
        """
        For this character asset find its name used in the command file.
        """ 
        split_path = prim_path.split("/")
        prim_name = split_path[-1]
        root_path = self.setting.get(PeopleSettings.CHARACTER_PRIM_PATH)
        # If a character is loaded through the spawn command, the commands for the character can be given by using the encompassing parent name.
        if prim_path.startswith(str(root_path)):
            parent_len = len(root_path.split("/"))
            parent_name = split_path[parent_len]
            return parent_name
        return prim_name
 
    def init_character(self):
        """
        Initializes global variables and fetches animation graph attached to the character. Called after entering runtime as ag.get_character() can only be used in runtime.
        """
        print("init_character")
        self.navigation_manager = NavigationManager(str(self.prim_path), self.navmeshEnabled, self.avoidanceOn)
        self.character_manager = GlobalCharacterPositionManager.get_instance()
        self.queue_manager = GlobalQueueManager.get_instance()
        if not self.navigation_manager or not self.character_manager or not self.queue_manager:
            return False

        self.character = ag.get_character(str(self.prim_path))
        if self.character is None:
            return False

        self.commands = self.get_simulation_commands()
        self.character.set_variable("Action", "None")
        carb.log_info("Initialize the character")
        return True

    def read_commands_from_file(self):
        """
        Reads commands from file pointed by self.command_path. Creates a Queue using queue manager if a queue is specified.
        :return: List of commands.
        :rtype: python list
        """
        result, version, context = omni.client.read_file(self.command_path)
        if result != omni.client.Result.OK:
            return []

        cmd_lines = memoryview(context).tobytes().decode("utf-8").splitlines()
        return cmd_lines


    def read_commands_from_UI(self):
        ui_commands = CommandTextWidget.textbox_commands
        print("UI Commands")
        print(ui_commands)
        if ui_commands:
            cmd_lines = ui_commands.splitlines()
            return cmd_lines
        return []


    def get_combined_user_commands(self):
        cmd_lines = []

        # Get commands from cmd_file
        cmd_lines.extend(self.read_commands_from_file())

        # Get commands from UI
        cmd_lines.extend(self.read_commands_from_UI())

        return cmd_lines


    def get_simulation_commands(self):
        cmd_lines = self.get_combined_user_commands()
        commands = []
        for cmd_line in cmd_lines:
            if not cmd_line:
                continue
            words = cmd_line.strip().split(' ')
            if words[0] == self.character_name:
                command = []
                for word in words[1:]:
                    command.append(word)
                commands.append(command)
            if words[0] == "Queue":
                self.queue_manager.create_queue(words[1])
            if words[0] == "Queue_Spot":
                queue = self.queue_manager.get_queue(words[1])
                queue.create_spot(int(words[2]), carb.Float3(float(words[3]),float(words[4]),float(words[5])), Utils.convert_angle_to_quatd(float(words[6])))
            if words[0][0] == "#":
                continue
        return commands


    def get_command(self, command):
        """
        Returns an instance of a command object based on the command.

        :param list[str] command: list of strings describing the command.
        :return: instance of a command object.
        :rtype: python object
        """
        print(command)
        if command[0] == "GoTo":
            return GoTo(self.character, command, self.navigation_manager)
        elif command[0] == "GoToLoop":
            return GoToLoop(self.character, command, self.navigation_manager) #DILLON: VLN
        elif command[0] == "GoToNew":
            return GoToNew(self.character, command, self.navigation_manager, self.current_time) #DILLON
        elif command[0] == "WaitTillTime":
            return WaitTillTime(self.character, command, self.navigation_manager, self.current_time) #DILLON
        elif command[0] == "Teleport":
            return Teleport(self.character, command, self.navigation_manager) #DILLON
        elif command[0] == "Idle":
            return Idle(self.character, command, self.navigation_manager)
        elif command[0] == "Queue":
            return QueueCmd(self.character, command, self.navigation_manager, self.queue_manager)
        elif command[0] == "Dequeue":
            return Dequeue(self.character, command, self.navigation_manager, self.queue_manager)
        elif command[0] == "LookAround":
            return LookAround(self.character, command, self.navigation_manager)
        elif command[0] == "Sit":
            return Sit(self.character, command, self.navigation_manager)
        else:
            module_str = ".commands.{}".format(command[0].lower(), package = None)
            try:
                custom_class = getattr(importlib.import_module(module_str, package=__package__), command[0])
            except (ImportError, AttributeError) as error:
                carb.log_error("Module or Class for the command do not exist. Check the command again.")
            return custom_class(self.character, command, self.navigation_manager)


    def execute_command(self, commands, delta_time):
        """
        Executes commands in commands list in sequence. Removes a command once completed.

        :param list[list] commands: list of commands.
        :param float delta_time: time elapsed since last execution.
        """
        if self.current_command is None:
            if commands:
                self.current_command = self.get_command(commands[0])
            else:
                return

        if self.current_command.execute(delta_time):
            commands.pop(0)
            self.current_command = None


    def on_update(self, current_time: float, delta_time: float):
        """
        Called on every update. Initializes character at start, publishes character positions and executes character commands.
        :param float current_time: current time in seconds.
        :param float delta_time: time elapsed since last update.
        """
        print("on_update")
        #print(current_time)
        #print(self.sum_time)
        #print(time.time() - self.sim_start_time)
        #self.sum_time = self.sum_time + delta_time
        self.current_time = current_time

        if self.character is None:
            if not self.init_character():
                return

        if self.avoidanceOn:
            self.navigation_manager.publish_character_positions(delta_time, 0.5)

        if self.commands:
            self.execute_command(self.commands, delta_time)