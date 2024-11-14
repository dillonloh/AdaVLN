import numpy as np

# Helper functions
def normalize_angle(angle):
    """Normalizes an angle to be within the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def handle_stop_command(jetbot, jetbot_controller, world):
    """Stops the robot and pauses the simulation."""
    print("Stop command received. Stopping robot and pausing simulation.")
    jetbot.apply_wheel_actions(jetbot_controller.forward([0.0, 0.0]))
    world.pause()

def handle_robot_move_command(initial_position, current_position, jetbot, jetbot_controller, linear_speed, move_distance):
    """Moves the robot forward until the specified move_distance is reached from the initial position."""
    distance_traveled = np.linalg.norm(current_position - initial_position)
    if distance_traveled >= move_distance:
        print(f"Reached target distance: {distance_traveled:.2f} meters.")
        return True
    else:
        throttle = linear_speed
        print(f"Moving forward: distance traveled = {distance_traveled:.2f} meters")
        jetbot.apply_wheel_actions(jetbot_controller.forward([throttle, 0.0]))
        return False

def handle_robot_turn_command(initial_yaw, current_yaw, turn_direction, jetbot, jetbot_controller, rotation_speed, rotation_angle):
    """Turns the robot until the specified rotation_angle is reached from the initial yaw."""
    yaw_diff = np.abs(normalize_angle(current_yaw - initial_yaw))
    if yaw_diff >= rotation_angle:
        print(f"Reached target rotation angle: {np.degrees(yaw_diff):.2f} degrees.")
        return True
    else:
        steering = rotation_speed if turn_direction == "left" else -rotation_speed
        print(f"Turning {turn_direction}: yaw difference = {np.degrees(yaw_diff):.2f} degrees")
        jetbot.apply_wheel_actions(jetbot_controller.forward([0.0, steering]))
        return False
