import pandas as pd
import math
from playhouse.sqlite_ext import *
import json
from tabulate import tabulate  # For LaTeX table generation

# Function to calculate the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Initialize database
db = SqliteExtDatabase('/home/dillon/0Research/VLNAgent/AdaVLN/database/people.db')

class WorldState(Model):
    scene_id = CharField()
    episode_id = CharField()
    sim_time = FloatField()
    robot_x = FloatField()
    robot_y = FloatField()
    robot_z = FloatField()
    robot_yaw = FloatField()
    characters = JSONField(null=True)  # JSON field to store dynamic numbers of characters
    collided_with_building = BooleanField()

    class Meta:
        database = db

# Connect to the database
db.connect()

# Initialize a list to store statistics for each episode
episode_statistics = []

for episode_id in range(1, 10):  # Loop through episodes 1 to 9
    # Query data for the current episode
    episode_data = (
        WorldState.select()
        .where(WorldState.episode_id == str(episode_id))
        .order_by(WorldState.sim_time)
    )
    
    total_navigation_time = 0.0
    environmental_collision_time = 0.0
    human_collision_time = 0.0

    # Process the data
    for state in episode_data:
        total_navigation_time += 1  # Assuming each record represents 1 second

        # Environmental collision
        if state.collided_with_building:
            environmental_collision_time += 1

        # Human collision
        characters = state.characters or []
        for char in characters:
            char_x, char_y = char.get('x', 0), char.get('y', 0)
            robot_x, robot_y = state.robot_x, state.robot_y
            distance = calculate_distance(robot_x, robot_y, char_x, char_y)
            if distance < 0.2:
                human_collision_time += 1
                break  # Only count collision once per time step

    # Calculate NC ratios
    environmental_nc = (
        environmental_collision_time / total_navigation_time if total_navigation_time > 0 else 0
    )
    human_nc = (
        human_collision_time / total_navigation_time if total_navigation_time > 0 else 0
    )
    combined_nc = environmental_nc + human_nc

    # Append episode statistics
    episode_statistics.append({
        'Episode': episode_id,
        'Environmental NC': round(environmental_nc, 2),
        'Human NC': round(human_nc, 2),
        'Combined NC': round(combined_nc, 2),
    })

# Calculate averages across all episodes
average_env_nc = round(
    sum(stat['Environmental NC'] for stat in episode_statistics) / len(episode_statistics), 2
)
average_human_nc = round(
    sum(stat['Human NC'] for stat in episode_statistics) / len(episode_statistics), 2
)
average_combined_nc = round(
    sum(stat['Combined NC'] for stat in episode_statistics) / len(episode_statistics), 2
)

# Add the averages to the statistics table
episode_statistics.append({
    'Episode': 'Average',
    'Environmental NC': average_env_nc,
    'Human NC': average_human_nc,
    'Combined NC': average_combined_nc,
})

# Convert to a Pandas DataFrame for display
df = pd.DataFrame(episode_statistics)

# Generate a LaTeX-compatible table
latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

# Save the LaTeX table to a file
with open('episode_statistics.tex', 'w') as f:
    f.write(latex_table)

# Display the table
print(df)
print("\nLaTeX Table:\n")
print(latex_table)

# Save the table to a CSV file
df.to_csv('episode_statistics.csv', index=False)

# Close the database connection
db.close()
