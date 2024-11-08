import os
from playhouse.sqlite_ext import *

db = SqliteExtDatabase(os.path.join(os.path.dirname(__file__), 'people.db'))

