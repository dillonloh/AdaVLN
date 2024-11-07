import os
from peewee import *

db = SqliteDatabase(os.path.join(os.path.dirname(__file__), 'people.db'))

