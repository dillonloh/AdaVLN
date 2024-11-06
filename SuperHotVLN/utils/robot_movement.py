import numpy as np

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def handle_stop_command(jetbot, jetbot_controller, world):
    print("Stop command received. Stopping robot and pausing simulation.")
    jetbot.apply_wheel_actions(jetbot_controller.forward([0.0, 0.0]))
    world.pause()

def handle_robot_move_command(current_position, current_yaw, target_position, jetbot, jetbot_controller, distance_threshold=0.1):
    if target_position is None:
        direction = np.array([np.cos(current_yaw), np.sin(current_yaw)])
        target_position = current_position + direction * 0.5
        print(f"Target position set: {target_position}")

    distance = np.linalg.norm(target_position - current_position)
    if distance <= distance_threshold:
        print("Reached target position.")
        return None
    else:
        throttle, steering = 0.5, 0
        print(f"Moving forward: distance to target = {distance}")
        jetbot.apply_wheel_actions(jetbot_controller.forward([throttle, steering]))
        return target_position

def handle_robot_turn_command(current_yaw, TURN_ANGLE, turn_direction, target_yaw, jetbot, jetbot_controller):
    if target_yaw is None:
        target_yaw = normalize_angle(current_yaw + TURN_ANGLE if turn_direction == "left" else current_yaw - TURN_ANGLE)
        print(f"Initial yaw: {current_yaw} radians, Target yaw set: {target_yaw} radians")
    
    yaw_diff = np.abs(current_yaw - target_yaw)
    if yaw_diff <= np.radians(1):
        print(f"Reached target yaw ({turn_direction}).")
        return None
    else:
        throttle, steering = 0, 0.5 if turn_direction == "left" else -1
        print(f"Turning {turn_direction}: yaw difference = {yaw_diff} radians")
        jetbot.apply_wheel_actions(jetbot_controller.forward([throttle, steering]))
        return target_yaw
