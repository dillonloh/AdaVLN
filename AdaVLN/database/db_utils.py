import os
from playhouse.sqlite_ext import *

def create_db():

    db = SqliteExtDatabase(os.path.join(os.path.dirname(__file__), 'people.db'))

    return db
