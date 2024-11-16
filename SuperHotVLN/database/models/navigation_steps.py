from playhouse.sqlite_ext import *

from ..db_utils import create_db

class NavigationSteps(Model):
    scene_id = CharField()
    episode_id = CharField()
    sim_start_time = FloatField()
    sim_end_time = FloatField(null=True)
    step_num = IntegerField()
    action = CharField()
    robot_x = FloatField()
    robot_y = FloatField()
    robot_z = FloatField()
    robot_yaw = FloatField()
    collided_with_building = BooleanField()

    class Meta:
        database = create_db()
