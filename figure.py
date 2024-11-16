import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from playhouse.sqlite_ext import *
import json

# Initialize database
db = SqliteExtDatabase('/home/dillon/0Research/VLNAgent/SuperHotVLN/database/people.db')

# Define models
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

# Connect to the database
db.connect()

# Fetch WorldState data for episode 1
world_states = list(WorldState.select().where(WorldState.episode_id == '1'))

# Load task details to fetch goal and start positions
task_details_path = "/home/dillon/0Research/VLNAgent/example_dataset/tasks/tasks.json"
with open(task_details_path, "r") as f:
    task_details_list = json.load(f).get("episodes")

episode_number = 9
current_task = task_details_list[episode_number - 1]

# Extract goal position, radius, and start position
goal_pos = current_task["goals"][0]["position"]
goal_radius = current_task["goals"][0]["radius"]
start_pos = current_task["start_position"]

# Prepare data for plotting
robot_positions = [(ws.robot_x, ws.robot_y) for ws in world_states]
character_positions = {}
for ws in world_states:
    if ws.characters:
        for character in ws.characters:
            char_id = character['character_id']
            pos = (character['pos_x'], character['pos_y'])
            if char_id not in character_positions:
                character_positions[char_id] = []
            character_positions[char_id].append(pos)

# Create the plot
plt.figure(figsize=(10, 10))

# Plot robot movement
robot_x, robot_y = zip(*robot_positions)
plt.plot(robot_x, robot_y, label='Robot Path', color='blue')

# Plot character movements with unique colors
colors = plt.cm.get_cmap('tab10', len(character_positions))
for idx, (char_id, positions) in enumerate(character_positions.items()):
    char_x, char_y = zip(*positions)
    plt.plot(char_x, char_y, label=f'Character {char_id}', color=colors(idx))

# Plot start position
plt.scatter(start_pos[0], start_pos[1], color='orange', label='Start Position', zorder=5)

# Plot goal point with a dotted circle
plt.scatter(*goal_pos, color='red', label='Goal Point', zorder=5)
circle = Circle((goal_pos[0], goal_pos[1]), goal_radius, color='red', fill=False, linestyle='dotted', label='Goal Radius')
plt.gca().add_patch(circle)

# Finalize plot
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Episode 1: Robot and Character Movements')
plt.grid()
plt.axis('equal')
plt.tight_layout()

# Show plot
plt.show()

# Close the database connection
db.close()
