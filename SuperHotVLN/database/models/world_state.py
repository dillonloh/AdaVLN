from playhouse.sqlite_ext import *

from ..db_utils import create_db

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
        database = create_db()
