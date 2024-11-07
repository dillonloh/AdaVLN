from peewee import *

from ..db_utils import db

class WorldState(Model):
    
    env_id = CharField()
    task_id = CharField()
    sim_time = FloatField()
    robot_x = FloatField()
    robot_y = FloatField()
    robot_z = FloatField()
    robot_yaw = FloatField()
    human_x = FloatField(null=True)  # Make these nullable if they might not always be set
    human_y = FloatField(null=True)
    human_z = FloatField(null=True)
    class Meta:
        database = db # This model uses the "people.db" database.