"""
Created on Wed Jan 27 20:46:04 2021

@author: colin
"""
from bookrecommender import Base

# just create tabel models on tables that already exist in DB
Book = Base.classes.books_meta
Rating = Base.classes.user_ratings


# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_name = db.Column(db.String(32), )
