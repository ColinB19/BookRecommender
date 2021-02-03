"""
Created on Wed Jan 27 20:46:04 2021

@author: colin
"""
from bookrecommender import Base

# just create tabel models on tables that already exist in DB
Book = Base.classes.books_meta
Rating = Base.classes.user_ratings