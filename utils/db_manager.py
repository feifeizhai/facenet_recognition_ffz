from peewee import *
import datetime


db = SqliteDatabase('face_data.db')

class BaseModel(Model):
    class Meta:
        database = db

class Person(BaseModel):
    user_name = TextField()
    real_name = TextField()
    created_date = DateTimeField(default=datetime.datetime.now)
    face_data = TextField()
    euc_dists = TextField()
    cos_dists = TextField()
    # vertices = TextField()

