import os
from playhouse.sqlite_ext import *

def create_db():
    # check if the database already exists
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'people.db')):
        # recreate the database
        print("Removing existing database...")
        os.remove(os.path.join(os.path.dirname(__file__), 'people.db'))

    db = SqliteExtDatabase(os.path.join(os.path.dirname(__file__), 'people.db'))

    return db
