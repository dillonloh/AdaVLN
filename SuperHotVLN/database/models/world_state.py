from playhouse.sqlite_ext import *

from ..db_utils import db

class WorldState(Model):
    env_id = CharField()
    task_id = CharField()
    sim_time = FloatField()
    robot_x = FloatField()
    robot_y = FloatField()
    robot_z = FloatField()
    robot_yaw = FloatField()
    characters = JSONField(null=True)  # JSON field to store dynamic numbers of characters

    class Meta:
        database = db
