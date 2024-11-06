## anim use
import json
import random
import time

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import VisualCuboid
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.kit.actions.core

import carb

from omni.anim.people.ui_components.command_setting_panel.command_text_widget import CommandTextWidget
from pxr import Sdf, Gf, UsdGeom
from omni.isaac.core.utils import prims


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

def generate_cmd_lines(humans_dict):
    """
    Takes in the humans dict in the task json and generates command lines for omni.anim.people use
    """

    full_cmd_lines = []
    i = 0
    for human in humans_dict:
        human_cmd_lines = [] 
        name = human.get("name")
        if not name:
            # random unused name
            name = f"Human_{i}"
            i += 1
        
        spawn = human.get("spawn")
        waypoints = human.get("waypoints")
        spawn_cmd = f"Spawn {name} {spawn[0]} {spawn[1]} {spawn[2]} {spawn[3]}"
        human_cmd_lines.append(spawn_cmd)
        
        gotoloop_cmd = f"{name} GoToLoop"
        for waypoint in waypoints:
            gotoloop_cmd += f" {waypoint[0]} {waypoint[1]} {waypoint[2]} _"

        human_cmd_lines.append(gotoloop_cmd)
    
        full_cmd_lines.extend(human_cmd_lines)

    return full_cmd_lines