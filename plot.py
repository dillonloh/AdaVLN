import json
import numpy as np
import matplotlib.pyplot as plt
from playhouse.sqlite_ext import *

task_details_path = "/home/hiverlab-workstation/Research/SuperHotVLN/example_dataset/tasks/tasks.json"
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
db = SqliteExtDatabase("/home/hiverlab-workstation/Research/SuperHotVLN/SuperHotVLN/database/people.db")

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

# Define the Statistics model to store the calculated statistics
class Statistics(Model):
    episode_id = CharField()
    success_rate = FloatField()
    oracle_success_rate = FloatField()
    spl = FloatField()
    navigation_error = FloatField()
    trajectory_length = FloatField()
    total_collision_rate = FloatField()
    dynamic_collision_rate = FloatField()

    class Meta:
        database = db

# Create tables if they don't exist
db.create_tables([Statistics])

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to generate statistics and plot

def generate_statistics_and_plot(episode_number):
    all_rows = WorldState.select().where(WorldState.episode_id == episode_number)

    total_distance = 0.0
    last_position = None
    robot_final_pos = None
    collision_count = 0
    dynamic_collision_count = 0
    oracle_success = False
    robot_positions = []
    human_positions = []

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(goal_pos[0], goal_pos[1], 'go', label='Goal Position', markersize=10)
    circle = plt.Circle((goal_pos[0], goal_pos[1]), goal_radius, color='g', fill=False, linestyle='--', label='Success Radius (3m)')
    ax.add_artist(circle)
    ax.plot(start_pos[0], start_pos[1], 'bs', label='Start Position', markersize=10)

    human_legend_added = False  # Flag to ensure one legend entry for humans

    for row in all_rows:
        robot_pos = np.array([row.robot_x, row.robot_y])
        robot_positions.append(robot_pos)
        ax.plot(robot_pos[0], robot_pos[1], 'b.', markersize=2)

        if last_position is not None:
            total_distance += calculate_distance(last_position, robot_pos)

        if row.characters:
            for character in row.characters:
                character_pos = np.array([character['pos_x'], character['pos_y']])
                human_positions.append(character_pos)

                # Add the legend only once for human positions
                if not human_legend_added:
                    ax.plot(character_pos[0], character_pos[1], 'r.', markersize=4, label='Human Position')
                    human_legend_added = True
                else:
                    ax.plot(character_pos[0], character_pos[1], 'r.', markersize=4)

                dist_to_character = calculate_distance(robot_pos, character_pos)
                if dist_to_character <= 1.0:
                    collision_count += 1
                if dist_to_character <= 0.5:
                    dynamic_collision_count += 1

        if calculate_distance(robot_pos, goal_pos[:2]) <= goal_radius:
            oracle_success = True

        last_position = robot_pos

    robot_final_pos = robot_positions[-1] if robot_positions else None
    final_distance_from_goal = calculate_distance(robot_final_pos, goal_pos[:2])
    final_success = (final_distance_from_goal <= goal_radius and collision_count == 0)
    spl = total_distance / max(total_distance, final_distance_from_goal)
    tcr = collision_count
    dcr = dynamic_collision_count

    # Save results to the database
    Statistics.create(
        episode_id=episode_number,
        success_rate=1.0 if final_success else 0.0,
        oracle_success_rate=1.0 if oracle_success else 0.0,
        spl=spl,
        navigation_error=final_distance_from_goal,
        trajectory_length=total_distance,
        total_collision_rate=tcr,
        dynamic_collision_rate=dcr,
    )

    # Plot settings and display
    min_x = min([pos[0] for pos in robot_positions + human_positions] + [goal_pos[0]] + [start_pos[0]]) - goal_radius - 1
    max_x = max([pos[0] for pos in robot_positions + human_positions] + [goal_pos[0]] + [start_pos[0]]) + goal_radius + 1
    min_y = min([pos[1] for pos in robot_positions + human_positions] + [goal_pos[1]] + [start_pos[1]]) - goal_radius - 1
    max_y = max([pos[1] for pos in robot_positions + human_positions] + [goal_pos[1]] + [start_pos[1]]) + goal_radius + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.set_title(f"Robot Path and Collisions for Episode {episode_number}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    stats_text = (
        f"Final Distance from Goal: {final_distance_from_goal:.2f} m\n"
        f"Total Distance: {total_distance:.2f} m\n"
        f"Collision Count: {collision_count}\n"
        f"Dynamic Collision Count: {dynamic_collision_count}\n"
        f"Final Success: {'Yes' if final_success else 'No'}\n"
        f"Oracle Success: {'Yes' if oracle_success else 'No'}\n"
        f"SPL: {spl:.2f}\n"
    )
    fig.text(0.1, 0.1, stats_text, fontsize=12, verticalalignment='center', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig(f"episode_{episode_number}_plot.png")

# Example usage
generate_statistics_and_plot(episode_number)
