import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from playhouse.sqlite_ext import *
import json
import math

# Initialize database
db = SqliteExtDatabase('/home/dillon/0Research/VLNAgent/SuperHotVLN/database/people.db')

class WorldState(Model):
    scene_id = CharField()
    episode_id = CharField()
    sim_time = FloatField()
    robot_x = FloatField()
    robot_y = FloatField()
    robot_z = FloatField()
    robot_yaw = FloatField()
    characters = JSONField(null=True)
    collided_with_building = BooleanField()

    class Meta:
        database = db

db.connect()

task_details_path = "/home/dillon/0Research/VLNAgent/example_dataset/tasks/tasks.json"
with open(task_details_path, "r") as f:
    task_details_list = json.load(f).get("episodes")

selected_episodes = [1, 4, 8]  # Example episodes

# Create a figure with a custom layout
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], figure=fig)

# Axes for each episode
ax1 = fig.add_subplot(gs[0, 0])  # Top left
ax2 = fig.add_subplot(gs[0, 1])  # Top right
ax3 = fig.add_subplot(gs[1, 0])  # Bottom left
legend_ax = fig.add_subplot(gs[1, 1])  # Bottom right for the legend

# Loop through selected episodes and assign axes
axes = [ax1, ax2, ax3]
for ax, episode_number in zip(axes, selected_episodes):
    # Fetch WorldState data
    world_states = list(WorldState.select().where(WorldState.episode_id == str(episode_number)))
    current_task = task_details_list[episode_number - 1]
    goal_pos = current_task["goals"][0]["position"]
    goal_radius = current_task["goals"][0]["radius"]
    start_pos = current_task["start_position"]
    # Flip start pos y and z
    start_pos[1], start_pos[2] = start_pos[2], start_pos[1]
    # Flip goal pos y and z
    goal_pos[1], goal_pos[2] = goal_pos[2], goal_pos[1]

    # Convert 3D to 2D positions
    start_pos_2d = start_pos[:2]
    goal_pos_2d = goal_pos[:2]

    # Prepare data
    robot_positions = [(ws.robot_x, ws.robot_y) for ws in world_states if ws.robot_x is not None and ws.robot_y is not None]
    character_positions = {}
    for ws in world_states:
        if ws.characters:
            for char in ws.characters:
                char_id = char['character_id']
                pos_x, pos_y = char.get('pos_x'), char.get('pos_y')
                if pos_x is not None and pos_y is not None:
                    if char_id not in character_positions:
                        character_positions[char_id] = []
                    character_positions[char_id].append((pos_x, pos_y))

    # Plot robot movement if data exists
    if robot_positions:
        robot_x, robot_y = zip(*robot_positions)
        ax.plot(robot_x, robot_y, label='Robot Path', color='blue', zorder=3)

    # Plot character movements using similar shades of the same color
    if character_positions:
        color_base = plt.cm.Blues
        for char_idx, (char_id, positions) in enumerate(character_positions.items()):
            if positions:
                char_x, char_y = zip(*positions)
                color = color_base(0.5 + char_idx * 0.25)  # Use shades of blue
                ax.plot(char_x, char_y, label=f'Character {char_id}', color=color, zorder=2)

    # Plot start position as a square
    if start_pos_2d:
        ax.scatter(start_pos_2d[0], start_pos_2d[1], color='orange', marker='s', label='Start Point', zorder=5)

    # Plot goal position as a triangle
    if goal_pos_2d:
        ax.scatter(goal_pos_2d[0], goal_pos_2d[1], color='red', marker='^', label='Goal Point', zorder=5)
        circle = Circle((goal_pos_2d[0], goal_pos_2d[1]), goal_radius, color='red', fill=False, linestyle='dotted', zorder=4)
        ax.add_patch(circle)

    ax.set_title(f'Episode {episode_number}', fontsize=16)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.grid(False)

# Hide gridlines for the legend axis and add legend
legend_ax.axis('off')
handles, labels = ax1.get_legend_handles_labels()
legend_ax.legend(handles, labels, loc='center', fontsize=18        )

plt.tight_layout()

# Save plot to avoid rendering issues
plt.savefig("output_plot.png", bbox_inches='tight')
plt.close(fig)

db.close()
