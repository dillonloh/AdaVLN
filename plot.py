import json
import numpy as np
import matplotlib.pyplot as plt
from playhouse.sqlite_ext import *

task_details_path = "/home/dillon/0Research/VLNAgent/example_dataset/tasks/GLAQ4DNUx5U.json"
with open(task_details_path, "r") as f:
    task_details_list = json.load(f).get("episodes")
episode_number = 1
current_task = task_details_list[episode_number - 1]
print(current_task)

# Assume goal_pos is a variable containing the goal position (x, y)
goal_pos = current_task["goals"][0]["position"]
goal_radius = current_task["goals"][0]["radius"]
start_pos = current_task["start_position"]

# Database connection
db = SqliteExtDatabase("/home/dillon/0Research/VLNAgent/SuperHotVLN/database/people.db")

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Define the WorldState model
class WorldState(Model):
    scene_id = CharField()
    episode_id = CharField()
    sim_time = FloatField()
    robot_x = FloatField()
    robot_y = FloatField()
    robot_z = FloatField()
    robot_yaw = FloatField()
    characters = JSONField(null=True)  # JSON field to store dynamic numbers of characters

    class Meta:
        database = db


# Function to generate the statistics and plot
def generate_statistics_and_plot(episode_number):
    # Fetch the rows for the given episode
    all_rows = WorldState.select().where(WorldState.episode_id == episode_number)

    # Initialize variables for statistics
    total_distance = 0.0
    last_position = None
    robot_final_pos = None
    collision = False
    oracle_success = False
    robot_positions = []
    human_positions = []

    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the goal position (only using X, Y coordinates)
    ax.plot(goal_pos[0], goal_pos[1], 'go', label='Goal Position', markersize=10)

    # Plot success radius (3m)
    circle = plt.Circle((goal_pos[0], goal_pos[1]), goal_radius, color='g', fill=False, linestyle='--', label='Success Radius (3m)')
    ax.add_artist(circle)

    # Plot the start position
    ax.plot(start_pos[0], start_pos[1], 'bs', label='Start Position', markersize=10)  # Blue square

    for row in all_rows:
        robot_pos = np.array([row.robot_x, row.robot_y])
        robot_positions.append(robot_pos)

        # Plot robot's trajectory
        ax.plot(robot_pos[0], robot_pos[1], 'b.', markersize=2)

        # Calculate total distance moved by the robot
        if last_position is not None:
            total_distance += calculate_distance(last_position, robot_pos)

        # Check if the robot is within 1 unit of any character (collision detection)
        if row.characters:
            for character in row.characters:
                character_pos = np.array([character['pos_x'], character['pos_y']])
                human_positions.append(character_pos)  # Store for later plotting
                if calculate_distance(robot_pos, character_pos) <= 1.0:
                    collision = True
                    # Plot collision area (human character radius)
                    ax.plot(character_pos[0], character_pos[1], 'rx', markersize=10, label='Collision Point')

        # Check if the robot is within the goal radius
        if calculate_distance(robot_pos, goal_pos[:2]) <= goal_radius:  # Only use X, Y for goal position
            oracle_success = True

        last_position = robot_pos  # Update the last position

    # Get final robot position
    if robot_positions:
        robot_final_pos = robot_positions[-1]

    # 1) Distance from goal: Compare robot final pos with goal pos
    final_distance_from_goal = calculate_distance(robot_final_pos, goal_pos[:2])  # Only use X, Y for goal position

    # 4) Final success: Robot is within 3 meters of the goal and no collisions
    final_success = (final_distance_from_goal <= goal_radius and not collision)

    # 5) Oracle success rate: Whether the robot was within 3m from goal at ANY point
    oracle_success_rate = oracle_success

    # Print the results
    print(f"Episode {episode_number} Statistics:")
    print(f"1) Final Distance from Goal: {final_distance_from_goal:.2f} meters")
    print(f"2) Total Distance Moved by Robot: {total_distance:.2f} meters")
    print(f"3) Collision with Character: {'Yes' if collision else 'No'}")
    print(f"4) Final Success (Within 3m from goal and no collisions): {'Yes' if final_success else 'No'}")
    print(f"5) Oracle Success Rate (Ever within 3m from goal): {'Yes' if oracle_success_rate else 'No'}")

    # Find the min and max coordinates for all relevant points
    min_x = min([pos[0] for pos in robot_positions] + [goal_pos[0]] + [start_pos[0]] + [pos[0] for pos in human_positions]) - goal_radius - 1
    max_x = max([pos[0] for pos in robot_positions] + [goal_pos[0]] + [start_pos[0]] + [pos[0] for pos in human_positions]) + goal_radius + 1
    min_y = min([pos[1] for pos in robot_positions] + [goal_pos[1]] + [start_pos[1]] + [pos[1] for pos in human_positions]) - goal_radius - 1
    max_y = max([pos[1] for pos in robot_positions] + [goal_pos[1]] + [start_pos[1]] + [pos[1] for pos in human_positions]) + goal_radius + 1

    # Set axis limits to ensure everything fits (including the radius and characters)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Make the plot square (equal aspect ratio)
    ax.set_aspect('equal')

    # Plot title and labels
    ax.set_title(f"Robot Path and Collisions for Episode {episode_number}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Move the legend outside of the plot to avoid overlap
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add the statistics as text outside the plot using fig.text()
    stats_text = (
        f"Final Distance from Goal: {final_distance_from_goal:.2f} m\n"
        f"Total Distance: {total_distance:.2f} m\n"
        f"Collision: {'Yes' if collision else 'No'}\n"
        f"Final Success: {'Yes' if final_success else 'No'}\n"
        f"Oracle Success Rate: {'Yes' if oracle_success_rate else 'No'}"
    )

    # Place the statistics text outside the plot area (in the right margin)
    fig.text(0.1, 0.2, stats_text, fontsize=12, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))

    # Adjust layout to ensure everything fits well
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
episode_number = 1  # Replace with the actual episode number you want to query
generate_statistics_and_plot(episode_number)
